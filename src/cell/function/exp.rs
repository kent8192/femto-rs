use super::Function;
use crate::cell::variable::Variable;
use ndarray::ArrayD;
use num::traits::{Float, Num, NumCast};

#[allow(dead_code)]
#[derive(Clone)]
struct Exp<T> {
    input: Option<Variable<ArrayD<T>>>,
}

impl<T> Exp<T> {
    #[allow(dead_code)]
    fn new() -> Self {
        Self { input: None }
    }
}

impl<T> Function<T> for Exp<T>
where
    T: Clone + Float + Num + NumCast + 'static,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        x.mapv(|e| e.exp())
    }

    fn backward(&mut self, gy: &ArrayD<T>) -> ArrayD<T> {
        let x = self.input.as_ref().unwrap().data.clone();
        let exp_x = x.mapv(|e| e.exp());
        exp_x * gy
    }

    fn remember_input(&mut self, input: &Variable<ArrayD<T>>) {
        self.input = Some(input.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{Exp, Function};
    use crate::cell::variable::Variable;
    use ndarray::ArrayD;

    #[test]
    fn it_works() {
        let mut exp = Exp::new();
        let input_data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let input = Variable::new(input_data);
        assert_eq!(
            exp.call(&input).data,
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 2]),
                vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
                    .into_iter()
                    .map(|e| e.exp())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        );
        let gy = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            exp.backward(&gy),
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 2]),
                vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64]
                    .into_iter()
                    .map(|e| e.exp())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        )
    }
}
