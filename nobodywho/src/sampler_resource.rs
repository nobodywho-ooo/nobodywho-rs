use crate::sampler_config;
use godot::prelude::*;

#[derive(GodotConvert, Var, Export, Debug, Clone, Copy)]
#[godot(via=GString)]
enum SamplerMethodName {
    Greedy,
    Temperature,
    MirostatV2,
    TopK,
}

#[derive(GodotClass)]
#[class(tool, base=Resource)]
pub struct NobodyWhoSampler {
    base: Base<Resource>,

    #[export]
    method: SamplerMethodName,

    pub sampler_config: sampler_config::SamplerConfig,
}

#[godot_api]
impl IResource for NobodyWhoSampler {
    fn init(base: Base<Resource>) -> Self {
        let methodname = match sampler_config::SamplerConfig::default().method {
            sampler_config::SamplerMethod::MirostatV2(_) => SamplerMethodName::MirostatV2,
            sampler_config::SamplerMethod::Temperature(_) => SamplerMethodName::Temperature,
            sampler_config::SamplerMethod::TopK(_) => SamplerMethodName::TopK,
            sampler_config::SamplerMethod::Greedy => SamplerMethodName::Greedy,
        };
        Self {
            method: methodname,
            sampler_config: sampler_config::SamplerConfig::default(),
            base,
        }
    }

    fn get_property_list(&mut self) -> Vec<godot::meta::PropertyInfo> {
        let base_properties = vec![];
        let penalty_properties = vec![
            godot::meta::PropertyInfo::new_export::<i32>("penalty_last_n"),
            godot::meta::PropertyInfo::new_export::<f32>("penalty_repeat"),
            godot::meta::PropertyInfo::new_export::<f32>("penalty_freq"),
            godot::meta::PropertyInfo::new_export::<f32>("penalty_present"),
            godot::meta::PropertyInfo::new_export::<bool>("penalize_nl"),
            godot::meta::PropertyInfo::new_export::<bool>("ignore_eos"),
        ];
        let method_properties = match self.method {
            SamplerMethodName::Greedy => vec![],
            SamplerMethodName::Temperature => vec![
                godot::meta::PropertyInfo::new_export::<u32>("seed"),
                godot::meta::PropertyInfo::new_export::<f32>("temperature"),
            ],
            SamplerMethodName::MirostatV2 => vec![
                godot::meta::PropertyInfo::new_export::<u32>("seed"),
                godot::meta::PropertyInfo::new_export::<f32>("temperature"),
                godot::meta::PropertyInfo::new_export::<f32>("tau"),
                godot::meta::PropertyInfo::new_export::<f32>("eta"),
            ],
            SamplerMethodName::TopK => vec![
                godot::meta::PropertyInfo::new_export::<u32>("seed"),
                godot::meta::PropertyInfo::new_export::<i32>("top_k"),
            ],
        };
        base_properties
            .into_iter()
            .chain(penalty_properties)
            .chain(method_properties)
            .collect()
    }

    fn get_property(&self, property: StringName) -> Option<Variant> {
        match (&self.sampler_config.method, property.to_string().as_str()) {
            (_, "method") => Some(Variant::from(self.method)),
            (_, "penalty_last_n") => Some(Variant::from(self.sampler_config.penalty_last_n)),
            (_, "penalty_repeat") => Some(Variant::from(self.sampler_config.penalty_repeat)),
            (_, "penalty_freq") => Some(Variant::from(self.sampler_config.penalty_freq)),
            (_, "penalty_present") => Some(Variant::from(self.sampler_config.penalty_present)),
            (_, "penalize_nl") => Some(Variant::from(self.sampler_config.penalize_nl)),
            (_, "ignore_eos") => Some(Variant::from(self.sampler_config.ignore_eos)),
            (sampler_config::SamplerMethod::Temperature(conf), "temperature") => {
                Some(Variant::from(conf.temperature))
            }
            (sampler_config::SamplerMethod::Temperature(conf), "seed") => {
                Some(Variant::from(conf.seed))
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "eta") => {
                Some(Variant::from(conf.eta))
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "tau") => {
                Some(Variant::from(conf.tau))
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "temperature") => {
                Some(Variant::from(conf.temperature))
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "seed") => {
                Some(Variant::from(conf.seed))
            }
            (sampler_config::SamplerMethod::TopK(conf), "top_k") => Some(Variant::from(conf.top_k)),
            (sampler_config::SamplerMethod::TopK(conf), "seed") => Some(Variant::from(conf.seed)),
            _ => {
                // self.base.to_gd().get_property()
                None
            } //panic!("Unexpected get property: {:?}", property)
        }
    }

    fn set_property(&mut self, property: StringName, value: Variant) -> bool {
        match (
            &mut self.sampler_config.method,
            property.to_string().as_str(),
        ) {
            (_, "method") => {
                let new_method = SamplerMethodName::try_from_variant(&value)
                    .expect("Unexpected: Got invalid sampler method");
                self.method = new_method;
                self.sampler_config.method = match new_method {
                    SamplerMethodName::Temperature => sampler_config::SamplerMethod::Temperature(
                        sampler_config::TemperatureConfig::default(),
                    ),
                    SamplerMethodName::MirostatV2 => sampler_config::SamplerMethod::MirostatV2(
                        sampler_config::MirostatV2Config::default(),
                    ),
                    SamplerMethodName::TopK => {
                        sampler_config::SamplerMethod::TopK(sampler_config::TopKConfig::default())
                    }
                    SamplerMethodName::Greedy => sampler_config::SamplerMethod::Greedy,
                };
                self.base
                    .to_gd()
                    .upcast::<Object>()
                    .notify_property_list_changed();
                return true;
            }
            (_, "penalty_last_n") => {
                self.sampler_config.penalty_last_n =
                    i32::try_from_variant(&value).expect("Unexpected type for penalty_last_n");
            }
            (_, "penalty_repeat") => {
                self.sampler_config.penalty_repeat =
                    f32::try_from_variant(&value).expect("Unexpected type for penalty_repeat");
            }
            (_, "penalty_freq") => {
                self.sampler_config.penalty_freq =
                    f32::try_from_variant(&value).expect("Unexpected type for penalty_freq");
            }
            (_, "penalty_present") => {
                self.sampler_config.penalty_present =
                    f32::try_from_variant(&value).expect("Unexpected type for penalty_present");
            }
            (_, "penalize_nl") => {
                self.sampler_config.penalize_nl =
                    bool::try_from_variant(&value).expect("Unexpected type for penalize_nl");
            }
            (_, "ignore_eos") => {
                self.sampler_config.ignore_eos =
                    bool::try_from_variant(&value).expect("Unexpected type for ignore_eos");
            }

            (sampler_config::SamplerMethod::Temperature(conf), "seed") => {
                conf.seed = u32::try_from_variant(&value).expect("Unexpected type for seed");
            }
            (sampler_config::SamplerMethod::Temperature(conf), "temperature") => {
                conf.temperature =
                    f32::try_from_variant(&value).expect("Unexpected type for temperature");
            }

            (sampler_config::SamplerMethod::MirostatV2(conf), "tau") => {
                conf.tau = f32::try_from_variant(&value).expect("Unexpected type for tau");
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "eta") => {
                conf.eta = f32::try_from_variant(&value).expect("Unexpected type for eta");
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "temperature") => {
                conf.temperature =
                    f32::try_from_variant(&value).expect("Unexpected type for temperature");
            }
            (sampler_config::SamplerMethod::MirostatV2(conf), "seed") => {
                conf.seed = u32::try_from_variant(&value).expect("Unexpected type for seed");
            }

            (sampler_config::SamplerMethod::TopK(conf), "top_k") => {
                conf.top_k = i32::try_from_variant(&value).expect("Unexpected type for top_k");
            }
            (sampler_config::SamplerMethod::TopK(conf), "seed") => {
                conf.seed = u32::try_from_variant(&value).expect("Unexpected type for seed");
            }
            _ => godot_warn!("Set unexpected property name: {:?}", property),
        }
        true
    }
}
