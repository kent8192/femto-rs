use super::Function;
use crate::cell::variable::Variable;
use ndarray::ArrayD;
use num::Float;

#[allow(dead_code)]
#[derive(Clone)]
struct Square<T> {
    input: Option<Variable<ArrayD<T>>>,
}

impl<T> Square<T> {
    #[allow(dead_code)]
    fn new() -> Self {
        Self { input: None }
    }
}

impl<T> Function<T> for Square<T>
where
    T: Clone + Float + ndarray::ScalarOperand,
{
    fn forward(&mut self, x: &ArrayD<T>) -> ArrayD<T> {
        x * x
    }

    fn backward(&mut self, gy: &ArrayD<T>) -> ArrayD<T> {
        let x = self.input.as_ref().unwrap().data.clone();
        let two = T::from(2.0).unwrap();
        gy * &x * two
    }

    fn remember_input(&mut self, input: &Variable<ArrayD<T>>) {
        self.input = Some(input.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::{Function, Square};
    use crate::cell::variable::Variable;
    use ndarray::ArrayD;

    #[test]
    fn call_works() {
        let mut square = Square::new();
        let input_data =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let input = Variable::new(input_data);
        let output = square.call(&input);
        assert_eq!(
            output.data,
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 4.0, 9.0, 16.0]).unwrap()
        );
        let gy = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            square.backward(&gy),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![2.0, 4.0, 6.0, 8.0]).unwrap()
        )
    }
}
