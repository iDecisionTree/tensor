use tensor::Tensor;

fn main(){
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let shape: Vec<usize> = vec![2, 447];
    let tensor: Tensor = Tensor::new(data, shape).unwrap();

    println!("{:?}", tensor.data);
}