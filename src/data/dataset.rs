use burn::data::dataset::Dataset;

pub static CLASS_NAMES: [&str; 5] = ["1", "2", "3", "4", "5"];

#[derive(Clone, Debug)]
pub struct DatasetItem
{
    pub text: String,
    pub label: u8,
}

pub struct YaDataset
{
    items: Vec<DatasetItem>,
}

impl YaDataset
{
    pub fn new() -> Self {
        Self {
            items: vec![],
        }
    }

    pub fn from_items(items: Vec<DatasetItem>) -> Self {
        Self {
            items
        }
    }
    pub fn fill(&mut self, path: &str, limit: usize)
    {
        let mut iteration = 0;
        let mut reader = csv::ReaderBuilder::new().delimiter(b';').from_path(path).unwrap();
        for record in reader.records() {
            if iteration > 0 && iteration == limit {
                return;
            }
            let r = record.unwrap();
            let text = r.get(0).unwrap().to_string();
            let rating: f32 = r.get(2).unwrap().parse().unwrap();
            self.items.push(DatasetItem {
                text,
                label: rating as u8,
            });
            iteration += 1;
        }
    }

    pub fn split(self, left: f32) -> (Self, Self) {
        let len = self.len();
        let split_at = (len as f32 * left).round() as usize;
        let (left, right) = self.items.split_at(split_at);
        (Self::from_items(left.to_vec()),
         Self::from_items(right.to_vec()))
    }
}

impl Dataset<DatasetItem> for YaDataset
{
    fn get(&self, index: usize) -> Option<DatasetItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}