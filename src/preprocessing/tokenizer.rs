const AVAILABLE_CHARS: [char; 32] = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п',
    'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч',
    'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я'];

pub struct Tokenizer
{}

impl Tokenizer
{
    pub fn encode(&self, data: String) -> Vec<String>
    {
        let mut result: Vec<String> = Vec::new();
        let splitted = data.split_whitespace();
        for s in splitted {
            let lowered = s.to_lowercase();
            let mut standartized = String::with_capacity(lowered.len());
            let mut last_char = '1';
            let mut chars = lowered.chars();
            for cb in chars {
                let mut c = cb;
                if c == 'ё' {
                    c = 'е';
                }
                if AVAILABLE_CHARS.contains(&c) {
                    if last_char != c {
                        last_char = c;
                        standartized.push(c);
                    }
                }
            }
            if standartized.len() > 3 {
                result.push(standartized);
            }
        }
        result
    }
    pub fn new() -> Self {
        Self {}
    }
}