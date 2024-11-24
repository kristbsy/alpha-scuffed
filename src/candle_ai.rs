use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

use crate::{
    game::{Game, Policy},
    model::TrainableModel,
};

const DEVICE: Device = Device::Cpu;

pub struct SimpleModel<const N: usize, const I: usize> {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    //varmap: VarMap,
    optimizer: candle_nn::AdamW,
}
/*
impl<const N: usize, const I: usize> SimpleModel<N, I> {
    pub fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_dim = 32;
        let l1 = linear(I, hidden_dim, vb.pp("layer 1"))?;
        let l2 = linear(hidden_dim, hidden_dim, vb.pp("layer 2"))?;
        let l3 = linear(hidden_dim, N, vb.pp("layer 3"))?;
        Ok(Self {
            layer1: l1,
            layer2: l2,
            layer3: l3,
        })
    }
}*/

impl<const N: usize, const I: usize> TrainableModel<N, I> for SimpleModel<N, I> {
    fn new() -> anyhow::Result<Self> {
        let hidden_dim = 32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &DEVICE);
        let optim_config = candle_nn::ParamsAdamW {
            lr: 1e-2,
            ..Default::default()
        };
        let l1 = linear(I, hidden_dim, vb.pp("layer 1"))?;
        let l2 = linear(hidden_dim, hidden_dim, vb.pp("layer 2"))?;
        let l3 = linear(hidden_dim, N, vb.pp("layer 3"))?;
        let optimizer = candle_nn::AdamW::new(varmap.all_vars(), optim_config)?;
        Ok(Self {
            layer1: l1,
            layer2: l2,
            layer3: l3,
            //varmap,
            optimizer,
        })
    }

    fn train(&mut self, dataset: crate::dataset::Dataset<N, I>) -> anyhow::Result<()> {
        const EPOCHS: usize = 100;
        let x = Tensor::from_vec(
            dataset.game_states.iter().cloned().flatten().collect(),
            (dataset.game_states.len(), I),
            &DEVICE,
        )?;
        let y = Tensor::from_vec(
            dataset.visit_stats.iter().cloned().flatten().collect(),
            (dataset.visit_stats.len(), N),
            &DEVICE,
        )?;
        eprintln!("x = {:#?}", x);
        eprintln!("y = {:#?}", y);
        for epoch in 0..EPOCHS {
            let output = self.forward(&x)?;
            let loss = candle_nn::loss::mse(&output, &y)?;
            self.optimizer.backward_step(&loss)?;
            if (epoch + 1) % 10 == 0 {
                println!("Train Loss: {}", loss.to_scalar::<f32>()?);
            }
        }
        Ok(())
    }

    fn predict(&self, state: [f32; I]) -> Result<([f32; N], f32), anyhow::Error> {
        let moves = self.predict_moves(state)?;
        let score = self.predict_score(state)?;
        Ok((moves, score))
    }

    fn predict_moves(&self, state: [f32; I]) -> anyhow::Result<[f32; N]> {
        let state_tensor = Tensor::from_slice(&state, (1, I), &DEVICE)?;
        let visits = self.forward(&state_tensor)?;
        let visits: Vec<f32> = visits.squeeze(0)?.to_vec1()?;
        let visits_array: [f32; N] = visits
            .try_into()
            .expect("wrong output dimension from visits prediction SimpleModel");
        Ok(visits_array)
    }

    fn predict_score(&self, state: [f32; I]) -> anyhow::Result<f32> {
        todo!()
    }
}

impl<const N: usize, const I: usize> Module for SimpleModel<N, I> {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.layer1.forward(xs)?;
        let x = x.relu()?;
        let x = self.layer2.forward(&x)?;
        let x = x.relu()?;
        let x = self.layer3.forward(&x)?;
        let x = candle_nn::ops::softmax(&x, 1)?;
        Ok(x)
    }
}

struct TemperaturedAiPolicy<const N: usize, const I: usize> {
    model: SimpleModel<N, I>,
}

impl<const N: usize, const I: usize, T: Game<N, I>> Policy<N, I, T> for TemperaturedAiPolicy<N, I> {
    fn select_move(&self, game: &T) -> anyhow::Result<usize> {
        todo!()
    }

    fn select_moves_batch(&self, games: Vec<&T>) -> anyhow::Result<Vec<usize>> {
        todo!()
    }
}

pub fn softmax<const N: usize>(data: Vec<[f32; N]>) -> anyhow::Result<Vec<[f32; N]>> {
    let mut out = Vec::new();
    let length = data.len();
    let flattened: Vec<_> = data.iter().cloned().flatten().collect();
    let tensor = Tensor::from_vec(flattened, (length, N), &DEVICE)?;
    let softmaxed = candle_nn::ops::softmax(&tensor, 1)?;

    for thing in softmaxed.flatten_all()?.to_vec1::<f32>()?.chunks_exact(N) {
        let mut pain = [0.0; N];
        pain[..N].copy_from_slice(&thing[..N]);
        out.push(pain);
    }

    Ok(out)
}
