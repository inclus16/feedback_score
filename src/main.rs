use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::process::exit;
use std::sync::Arc;
use std::time::Instant;

use burn::backend::{Autodiff, Fusion, Wgpu};
use burn::backend::wgpu::{AutoGraphicsApi, GraphicsApi, WgpuDevice};
use burn::config::Config;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::data::dataset::transform::{SamplerDataset, ShuffledDataset};
use burn::lr_scheduler::noam::NoamLrSchedulerConfig;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, Recorder, SensitiveCompactRecorder};
use burn::train::{LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition};
use burn::train::metric::{AccuracyMetric, CUDAMetric, LearningRateMetric, LossMetric};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn_tensor::{Data, Float, Tensor};
use burn_tensor::backend::{AutodiffBackend, Backend};
use csv::Position;
use indicatif::ProgressBar;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::data::csv_batcher::{TrainBatcher, ValBatcher};
use crate::data::dataset::{CLASS_NAMES, YaDataset};
use crate::models::linear::{LinearModel, LinearModelConfig};
use crate::preprocessing::tfidf::TfIdfVectorizer;
use crate::preprocessing::tokenizer::Tokenizer;

mod models;
mod preprocessing;
mod tests;
mod data;

type MyBackend = Fusion<Wgpu<AutoGraphicsApi, f32, i32>>;
type MyAutodiffBackend = Autodiff<Fusion<Wgpu<AutoGraphicsApi, f32, i32>>>;

#[derive(Config)]
pub struct LearnConfig {
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn main() {
    let artifact_dir = "artifacts";
    if Path::new(artifact_dir).exists() {
        let record = SensitiveCompactRecorder::new().load(format!("{artifact_dir}/model").into());
        let vectorizer = get_vectorizer();
        let model = LinearModelConfig::new(vectorizer.get_tokens_len()).init_with::<MyBackend>(record.unwrap())
            .to_device(&WgpuDevice::default());
        test(&model, &vectorizer);
    } else {
        learn::<MyAutodiffBackend>(WgpuDevice::default(), artifact_dir);
    }
}

pub fn test<B: Backend>(model: &LinearModel<B>, vectorizer: &TfIdfVectorizer)
{
    let mut textes: HashMap<String, u8> = HashMap::new();
    textes.insert(String::from("Сервис хороший , удобный , рядом с моим домом , работники вежливые , добрые , заехал за тонировать мне все сделали очень быстро , мне понравилось .Побольше таких людей таких добрых и приятных"), 5);
    textes.insert(String::from("Снять старую магнитолу и поставить на ее место новую просят 1300₽. Цена явно завышена в 2 раза. Не советую эту контору!"), 1);
    textes.insert(String::from("Много раз был в этом сервисе и ни чего плохого не сказал бы, но последний раз когда забирал авто сразу обратил внимание что на двери появилась небольшая вмятинка с парой сколов, говорить им как понял об этом нет смысла - ни кто не сознается"), 3);
    textes.insert(String::from("Отличный сервис"), 5);
    textes.insert(String::from("При установки сигнализации. Пацарапали лобовое, вот что делать сразу не увидел с утра на солнце сразу заметел, ещё какие то левые заманухи по телефону были"), 2);
    textes.insert(String::from("Низкое качество услуг при высоких ценах. Шараш монтаж.Не рекомендую."), 1);
    textes.insert(String::from("Обманывают клиентов по поводу стоимости.Приехал купить чехол для брелка Starline, нашёл нужный, спрашиваю продавца «Сколько стоит?». Ответ: «150 руб». Ок, без проблем, достаю дисконтную карту и уточняю, есть ли по ним скидки. Продавец замялся: «скидки по дисконтным картам... да, есть»"), 1);
    textes.insert(String::from("Отличный сервис, делал ремонт усилителя , цена отличная . Все установили и переделали некоторые моменты в проводке, плюс сделали сигнализацию. Всем советую, отзывчивый и добрый персонал ."), 5);
    textes.insert(String::from("Неплохой сервис, тоже посоветовал друган. Была проблема в электрике, подгнила проводка. Сделали неплохо. Ценой я вполне доволен. Все выполнено вполне аккуратно."), 4);
    for (text, label) in textes.iter() {
        let start = Instant::now();
        let predicted = predict(text, &vectorizer, &model);
        println!("Время предсказания: {:?} ms, Оценка сети: {}, реальная: {}, текст: {}", Instant::now().duration_since(start).as_millis(), predicted, label, text);
    }
}


pub fn learn<B: AutodiffBackend>(device: WgpuDevice, artifact_dir: &str)
{
    let vectorizer = Arc::new(get_vectorizer());
    let config = LearnConfig {
        optimizer: AdamConfig::new(),
        batch_size: 32,
        num_workers: 4,
        seed: 42,
        num_epochs: 3,
        learning_rate: 0.1,
    };
    B::seed(config.seed);

    let model: LinearModel<MyAutodiffBackend> = LinearModelConfig::new(vectorizer.get_tokens_len()).init();
    let batcher_train = TrainBatcher::<MyAutodiffBackend>::new(device.clone(), vectorizer.clone());
    let batcher_val = ValBatcher::<MyBackend>::new(device.clone(), vectorizer.clone());
    let mut dataset = YaDataset::new();
    dataset.fill("dataset.csv", 500000);
    let (train_ds, val_ds) = dataset.split(0.8);
    let mut rng = StdRng::from_entropy();
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(ShuffledDataset::new(train_ds, &mut rng));
    let val_size = val_ds.len();

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(SamplerDataset::new(val_ds, val_size));

    let lr_scheduler = NoamLrSchedulerConfig::new(config.learning_rate)
        .with_warmup_steps(1000)
        .init();

    let mut learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(SensitiveCompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<AccuracyMetric<MyAutodiffBackend>>(Aggregate::Mean, Direction::Highest, Split::Valid, StoppingCondition::NoImprovementSince {
            n_epochs: 3
        }))
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_val);
    test(&model_trained, &vectorizer);
    SensitiveCompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}

fn predict<B: Backend>(text: &str, vectorizer: &TfIdfVectorizer, model: &LinearModel<B>) -> String
{
    let text = vectorizer.process(text.to_string());
    let x = Tensor::from_floats(Data::from(text.as_slice())).reshape([1, text.len()]);
    let result = model.infer(x);
    get_class(result)
}

fn get_class<B: Backend>(predictions: Tensor<B, 2, Float>) -> String
{
    let prediction = predictions.clone();
    let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0];
    CLASS_NAMES[class_index as usize].to_string()
}

fn get_vectorizer() -> TfIdfVectorizer
{
    let tokenizer = Tokenizer::new();
    let mut vectorizer = TfIdfVectorizer::new(tokenizer);
    if Path::new("idf").exists() {
        println!("Idf dictionary found. Process with load...");
        vectorizer.load("idf");
    } else {
        println!("Idf was not found. Will make new");
        let mut reader = csv::ReaderBuilder::new().delimiter(b';').from_path("dataset.csv").unwrap();
        let mut total_count = 0;
        for record in reader.records() {
            total_count += 1;
        }
        println!("Total rows:{:?}", total_count);
        let pb = ProgressBar::new(total_count);
        let mut pos = Position::new();
        pos.set_line(1);
        reader.seek(pos).unwrap();
        for record in reader.records() {
            let record = record.unwrap();
            vectorizer.remember_one(record.get(0).unwrap().to_string());
            pb.inc(1);
        }
        pb.finish();
        vectorizer.sort();
        vectorizer.cut(10);
        vectorizer.save("idf");
    }
    vectorizer
}
