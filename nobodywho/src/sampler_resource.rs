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

macro_rules! property_list {
    ($self:expr,
     base: {$($base_field:ident : $base_type:ty),*},
     methods: {$($variant:ident { $($field:ident : $type:ty),*}),*}
     ) => {
        {
            let base_properties = vec![
                $(
                    godot::meta::PropertyInfo::new_export::<$base_type>(stringify!($base_field)),
                )*
            ];
            let method_properties = match $self.method {
                $(
                    SamplerMethodName::$variant => vec![
                        $(
                            godot::meta::PropertyInfo::new_export::<$type>(stringify!($field)),
                        )*
                    ],
                )*
            };
            let mut result: Vec<godot::meta::PropertyInfo> = base_properties;
            result.extend(method_properties);
            result
        }
    };
}

macro_rules! get_property {
    ($self:expr,
     $property:expr,
     base: {$($base_field:ident : $base_type:ty),*},
     methods: {$($variant:ident { $($variant_field:ident : $variant_type:ty),*}),*}
     ) => {{
        match (&$self.sampler_config.method, $property.to_string().as_str()) {
            (_, "method") => Some(Variant::from($self.method)),
            $(
                (_, stringify!($base_field)) => Some(Variant::from($self.sampler_config.$base_field)),
            )*
            $(
                // makes patterns like this:
                // (SamplerMethod::TopK(conf), "top_k") => Some(Variant::from(config.top_k))
                $(
                    (sampler_config::SamplerMethod::$variant(conf), stringify!($variant_field)) => Some(Variant::from(conf.$variant_field)),
                )*
            )*
            _ => None
        }
    }};
}

macro_rules! set_property {
    ($self:expr,
     $property:expr,
     $value:expr,
     base: {$($base_field:ident : $base_type:ty),*},
     methods: {$($variant:ident { $($variant_field:ident : $variant_type:ty),*}),*}
     ) => {{
        match (&mut $self.sampler_config.method, $property.to_string().as_str()) {
            (_, "method") => {
                let new_method = SamplerMethodName::try_from_variant(&$value).expect("Unexpected: Got invalid sampler method");
                $self.method = new_method;
                $self.sampler_config.method = match new_method {
                    $(
                        SamplerMethodName::$variant => {
                            sampler_config::SamplerMethod::$variant(sampler_config::$variant::default())
                        }
                    )*
                };
                $self.base
                    .to_gd()
                    .upcast::<Object>()
                    .notify_property_list_changed();
            },
            $(
                // generates arms like this:
                //     (_, "penalty_last_n") => {
                //         self.sampler_config.penalty_last_n =
                //             i32::try_from_variant(&value).expect("Unexpected type for penalty_last_n");
                (_, stringify!($base_field)) => {
                    $self.sampler_config.$base_field = <$base_type>::try_from_variant(&$value).expect(format!("Unexpected type for {}", stringify!($base_field)).as_str());
                }
            )*
            $(
                // generates ams like this:
                //     (sampler_config::SamplerMethod::Temperature(conf), "seed") => {
                //         conf.seed = u32::try_from_variant(&value).expect("Unexpected type for seed");
                //     }
                $(
                    (sampler_config::SamplerMethod::$variant(conf), stringify!($variant_field)) => {
                        conf.$variant_field = <$variant_type>::try_from_variant(&$value).expect(format!("Unexpected type for {}", stringify!($variant_field)).as_str());
                    }
                )*
            )*
            (variant, field_name) => unreachable!("Bad combination of method variant and property name: {:?} {:?}", variant, field_name),
        }
        true
    }};
}

#[godot_api]
impl IResource for NobodyWhoSampler {
    fn init(base: Base<Resource>) -> Self {
        let methodname = match sampler_config::SamplerConfig::default().method {
            sampler_config::SamplerMethod::MirostatV2(_) => SamplerMethodName::MirostatV2,
            sampler_config::SamplerMethod::Temperature(_) => SamplerMethodName::Temperature,
            sampler_config::SamplerMethod::TopK(_) => SamplerMethodName::TopK,
            sampler_config::SamplerMethod::Greedy(_) => SamplerMethodName::Greedy,
        };
        Self {
            method: methodname,
            sampler_config: sampler_config::SamplerConfig::default(),
            base,
        }
    }

    fn get_property_list(&mut self) -> Vec<godot::meta::PropertyInfo> {
        property_list!(
            self,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                penalize_nl: bool,
                ignore_eos: bool
            },
            methods: {
                Temperature { temperature: f32, seed: u32 },
                MirostatV2 { temperature: f32, seed: u32, tau: f32, eta: f32 },
                TopK { top_k: i32, seed: u32 },
                Greedy { }
            }
        )
    }

    fn get_property(&self, property: StringName) -> Option<Variant> {
        get_property!(
            self, property,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                penalize_nl: bool,
                ignore_eos: bool
            },
            methods: {
                Temperature { temperature: f32, seed: u32 },
                MirostatV2 { temperature: f32, seed: u32, tau: f32, eta: f32 },
                TopK { top_k: i32, seed: u32 },
                Greedy { }
            }
        )
    }

    fn set_property(&mut self, property: StringName, value: Variant) -> bool {
        set_property!(
            self, property, value,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                penalize_nl: bool,
                ignore_eos: bool
            },
            methods: {
                Temperature { temperature: f32, seed: u32 },
                MirostatV2 { temperature: f32, seed: u32, tau: f32, eta: f32 },
                TopK { top_k: i32, seed: u32 },
                Greedy { }
            }
        )
    }
}
