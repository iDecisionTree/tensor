use std::usize;

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor{
    pub fn new(data:Vec<f32>, shape:Vec<usize>) -> Result<Self,String>{
        let total_size = shape.iter().product();
        if data.len() != total_size{
            return Err(format!(
                "数据长度 {} 与形状 {:?} 不匹配（总大小：{}）",
                data.len(),
                shape,
                total_size,
            ));
        }

        return Ok(Tensor{
            data,
            shape,
        });
    }
}
