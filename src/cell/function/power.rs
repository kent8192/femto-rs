use super::Function;
use crate::cell::variable::Variable;
use ndarray::{ArrayD, ScalarOperand};
use num::traits::{Float, Num, NumCast};

#[allow(dead_code)]
#[derive(Clone)]
struct Power<T> {
    input: Option<Variable<ArrayD<T>>>,
    n: usize,
}

impl<T> Power<T> {
    #[allow(dead_code)]
    fn new(n: usize) -> Self {
        Self { input: None, n: n }
    }
}

impl<T> Function<T> for Power<T>
where
    T: Clone + Float + Num + NumCast + ScalarOperand + 'static,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        x.mapv(|e| e.powi(self.n as i32))
    }

    fn backward(&mut self, gy: &ArrayD<T>) -> ArrayD<T> {
        let x = self.input.as_ref().unwrap().data.clone();
        let n_minus_1 = T::from(self.n as f64 - 1.0).unwrap();
        let grad = x.mapv(|e| e.powf(n_minus_1));
        gy * grad * T::from(self.n as f64).unwrap()
    }

    fn remember_input(&mut self, input: &Variable<ArrayD<T>>) {
        self.input = Some(input.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{Function, Power};
    use crate::cell::variable::Variable;
    use ndarray::ArrayD;

    #[test]
    fn call_works() {
        let mut square = Power::new(5);
        let input_data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let input = Variable::new(input_data);
        let output = square.call(&input);
        assert_eq!(
            output.data,
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 32.0, 243.0, 1024.0])
                .unwrap()
        );
        let gy = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            square.backward(&gy),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![5.0, 80.0, 405.0, 1280.0])
                .unwrap()
        )
    }
}
