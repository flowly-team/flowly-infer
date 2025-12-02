#![allow(clippy::manual_retain)]

use std::sync::atomic::{AtomicU64, Ordering};

use futures::{Stream, TryStreamExt};

pub mod bbox;
mod error;
pub mod model;
pub mod nvinfer;
pub mod tract;

pub use error::Error;

use crate::model::{Either, Model};

pub struct ExecutionTask<T> {
    pub(crate) rx: tokio::sync::mpsc::Receiver<Result<T, Error>>,

    #[allow(dead_code)]
    pub(crate) handler: tokio::task::JoinHandle<()>,
}

impl<T> Stream for ExecutionTask<T> {
    type Item = Result<T, Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationKind {
    #[default]
    Float32,
    Float16,
    Int8,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationKind {
    #[default]
    Float32,
    Float16,
    Int8,
}

impl std::fmt::Display for QuantizationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                QuantizationKind::Float32 => "fp32",
                QuantizationKind::Float16 => "fp16",
                QuantizationKind::Int8 => "int8",
            }
        )
    }
}

impl std::str::FromStr for QuantizationKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" | "fp32" | "F32" | "FP32" => Ok(Self::Float32),
            "f16" | "fp16" | "F16" | "FP16" => Ok(Self::Float16),
            "int8" | "INT8" => Ok(Self::Int8),
            q => Err(Error::UnknownQuantizationMode(q.to_string())),
        }
    }
}

pub struct PeriodicCounter {
    count: AtomicU64,
    ts: AtomicU64,
    period: f64,
}

impl PeriodicCounter {
    pub const fn new(period: f64) -> Self {
        Self {
            period,
            count: AtomicU64::new(0),
            ts: AtomicU64::new(0),
        }
    }

    pub fn inc(&self) -> Option<f64> {
        let inst = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        if inst - f64::from_bits(self.ts.load(Ordering::SeqCst)) >= self.period {
            let val = self.count.load(Ordering::SeqCst) as f64 / self.period;

            self.ts.store(inst.to_bits(), Ordering::SeqCst);
            self.count.store(1, Ordering::SeqCst);

            Some(val)
        } else {
            self.count.fetch_add(1, Ordering::SeqCst);
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Projection {
    pub m: nalgebra::Matrix3<f32>,
}

pub trait AsVal<T> {
    fn as_val(&self) -> T;
}

#[derive(Debug, Clone)]
pub struct Bestshot<D> {
    pub inner: D,
    pub bestshot_score: f32,
}

pub trait Engine<In> {
    type Out: Send + 'static;

    fn infer(
        &self,
        input: impl Stream<Item = Result<In, Error>> + Send + 'static,
    ) -> impl Future<Output = Result<impl Stream<Item = Result<Self::Out, Error>> + Send + 'static, Error>>;
}

impl<I, L: Engine<I>, R: Engine<I>> Engine<I> for Either<L, R> {
    type Out = Either<L::Out, R::Out>;

    async fn infer(
        &self,
        input: impl Stream<Item = Result<I, Error>> + Send + 'static,
    ) -> Result<impl Stream<Item = Result<Self::Out, Error>> + Send + 'static, Error> {
        Ok(match self {
            Either::Left(e) => Either::Left(e.infer(input).await?.map_ok(Either::Left)),
            Either::Right(e) => Either::Right(e.infer(input).await?.map_ok(Either::Right)),
        })
    }
}

impl<In, E1, E2> Engine<In> for (E1, E2)
where
    E1: Engine<In>,
    E2: Engine<E1::Out>,
{
    type Out = E2::Out;
    async fn infer(
        &self,
        input: impl Stream<Item = Result<In, Error>> + Send + 'static,
    ) -> Result<impl Stream<Item = Result<Self::Out, Error>> + Send + 'static, Error> {
        let stream = self.0.infer(input).await?;
        let stream = self.1.infer(stream).await?;
        Ok(stream)
    }
}
