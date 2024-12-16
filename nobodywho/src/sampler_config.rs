use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

#[derive(Clone)]
pub struct SamplerConfig {
    pub method: SamplerMethod,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalize_nl: bool,
    pub ignore_eos: bool,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            penalty_last_n: -1,
            penalty_repeat: 0.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalize_nl: false,
            ignore_eos: false,
            method: SamplerMethod::Temperature(TemperatureConfig::default()),
            // method : SamplerMethod::MirostatV2(MirostatV2Config {
            //     seed: 1234,
            //     temperature: 0.8,
            //     tau: 5.0,
            //     eta: 0.1,
            // }),
        }
    }
}

#[derive(Clone)]
pub enum SamplerMethod {
    MirostatV2(MirostatV2Config),
    Temperature(TemperatureConfig),
    TopK(TopKConfig),
    Greedy,
}

#[derive(Clone)]
pub struct TopKConfig {
    pub top_k: i32,
    pub seed: u32,
}

impl Default for TopKConfig {
    fn default() -> Self {
        Self {
            top_k: 40,
            seed: 1234,
        }
    }
}

#[derive(Clone)]
pub struct MirostatV2Config {
    pub seed: u32,
    pub temperature: f32,
    pub tau: f32,
    pub eta: f32,
}

impl Default for MirostatV2Config {
    fn default() -> Self {
        Self {
            seed: 1234,
            temperature: 0.8,
            tau: 5.0,
            eta: 0.1,
        }
    }
}

#[derive(Clone)]
pub struct TemperatureConfig {
    pub seed: u32,
    pub temperature: f32,
}

impl Default for TemperatureConfig {
    fn default() -> Self {
        Self {
            seed: 1234,
            temperature: 0.8,
        }
    }
}

pub fn make_sampler(model: &LlamaModel, sampler_config: SamplerConfig) -> LlamaSampler {
    // init mirostat sampler
    let penalties = LlamaSampler::penalties(
        model.n_vocab(),
        model.token_eos().0,
        model.token_nl().0,
        sampler_config.penalty_last_n,
        sampler_config.penalty_repeat,
        sampler_config.penalty_freq,
        sampler_config.penalty_present,
        sampler_config.penalize_nl,
        sampler_config.ignore_eos,
    );
    let chainvec = match sampler_config.method {
        SamplerMethod::MirostatV2(conf) => {
            vec![
                penalties,
                LlamaSampler::temp(conf.temperature),
                LlamaSampler::mirostat_v2(conf.seed, conf.tau, conf.eta),
            ]
        }
        SamplerMethod::Temperature(conf) => {
            vec![
                penalties,
                LlamaSampler::temp(conf.temperature),
                LlamaSampler::dist(conf.seed),
            ]
        }
        SamplerMethod::Greedy => {
            vec![penalties, LlamaSampler::greedy()]
        }
        SamplerMethod::TopK(conf) => {
            vec![
                penalties,
                LlamaSampler::top_k(conf.top_k),
                LlamaSampler::dist(conf.seed),
            ]
        }
    };
    LlamaSampler::chain(chainvec, true)
}
