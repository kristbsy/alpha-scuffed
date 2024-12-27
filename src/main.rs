use crate::mcts::mcts;
use candle_ai::SimpleModel;
use checkers::Checkers;
use dataset::{create_dataset, save_dataset};
use game::{Game, Policy, RandomPolicy};
use hex::Hex;
use model::{AiPolicy, TrainableModel};

use std::fmt::Display;
mod candle_ai;
mod checkers;
mod dataset;
mod game;
mod hex;
mod mcts;
mod model;

fn play_games<const N: usize, const I: usize, T: Game<N, I> + Display, U: Policy<N, I, T>>(
    num_games: usize,
    policy: U,
) -> anyhow::Result<()> {
    for _ in 0..num_games {
        let mut game = T::new();
        println!("{game}");
        while !game.game_ended() {
            let next_move = policy.select_move(&game)?;
            game.perform_move(next_move);
            println!("{game}");
        }
    }
    Ok(())
}

fn training_loop<
    const N: usize,
    const I: usize,
    T: Game<N, I> + Display,
    M: TrainableModel<N, I>,
>(
    generations: usize,
) -> anyhow::Result<()> {
    let mut dataset = create_dataset::<N, I, T, RandomPolicy>(100, RandomPolicy {}, 0)?;
    save_dataset(&dataset.clone().into(), String::from("initial_dataset"));
    for generation in 0..generations {
        let mut model: M = M::new()?;
        model.train(dataset)?;
        // TODO: save model
        let policy = AiPolicy::<N, I, M> { model };
        dataset = create_dataset::<N, I, T, AiPolicy<N, I, M>>(50, policy, generation)?;
        save_dataset(
            &dataset.clone().into(),
            format!("generation_{}", generation),
        );
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    //play_games::<25, 50, Hex<25, 50>, RandomPolicy>(1000, RandomPolicy {})
    //training_loop::<25, 50, Hex<25, 50>>(1)
    const N: usize = 64;
    const I: usize = N * 2;
    training_loop::<N, I, Hex<N, I>, SimpleModel<N, I>>(10)
}
