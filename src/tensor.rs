use std::fmt;
use std::{
    ops::{Index, IndexMut},
    usize, vec,
};

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let total_size = shape.iter().product();
        if data.len() != total_size {
            return Err(format!(
                "数据长度 {} 与形状 {:?} 不匹配（总大小：{}）",
                data.len(),
                shape,
                total_size,
            ));
        }

        let strides = Self::calculate_strides(&shape);

        return Ok(Tensor {
            data: data,
            shape: shape,
            strides: strides,
        });
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, String> {
        let total_size = shape.iter().product();
        let data = vec![0.0; total_size];
        let strides = Self::calculate_strides(&shape);

        return Ok(Tensor {
            data: data,
            shape: shape,
            strides: strides,
        });
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self, String> {
        let total_size = shape.iter().product();
        let data = vec![1.0; total_size];
        let strides = Self::calculate_strides(&shape);

        return Ok(Tensor {
            data: data,
            shape: shape,
            strides: strides,
        });
    }

    pub fn full(shape: Vec<usize>, value: f32) -> Result<Self, String> {
        let total_size = shape.iter().product();
        let data = vec![value; total_size];
        let strides = Self::calculate_strides(&shape);

        return Ok(Tensor {
            data: data,
            shape: shape,
            strides: strides,
        });
    }

    pub fn rank(&self) -> Result<usize, String> {
        return Ok(self.shape.len());
    }

    pub fn numel(&self) -> Result<usize, String> {
        return Ok(self.data.len());
    }

    pub fn shape(&self) -> Result<&[usize], String> {
        return Ok(&self.shape);
    }

    pub fn data(&self) -> Result<&[f32], String> {
        return Ok(&self.data);
    }

    pub fn data_mut(&mut self) -> Result<&mut [f32], String> {
        return Ok(&mut self.data);
    }

    pub fn get(&self, indices: &[usize]) -> Result<&f32, String> {
        if indices.len() != self.rank().unwrap() {
            return Err(format!(
                "索引维度 {} 与张量秩 {} 不匹配",
                indices.len(),
                self.rank().unwrap()
            ));
        }
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[dim] {
                return Err(format!(
                    "索引 {} 超出张量维度 {} 的范围（大小：{}）",
                    idx, dim, self.shape[dim]
                ));
            }
        }

        let index = self.calculate_index(&indices);
        return Ok(self.data.get(index).unwrap());
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut f32, String> {
        if indices.len() != self.rank().unwrap() {
            return Err(format!(
                "索引维度 {} 与张量秩 {} 不匹配",
                indices.len(),
                self.rank().unwrap()
            ));
        }
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[dim] {
                return Err(format!(
                    "索引 {} 超出张量维度 {} 的范围（大小：{}）",
                    idx, dim, self.shape[dim]
                ));
            }
        }

        let index = self.calculate_index(&indices);
        return Ok(self.data.get_mut(index).unwrap());
    }

    pub fn set(&mut self, indices: &[usize], value: f32) -> Result<(), String> {
        if indices.len() != self.rank().unwrap() {
            return Err(format!(
                "索引维度 {} 与张量秩 {} 不匹配",
                indices.len(),
                self.rank().unwrap()
            ));
        }
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[dim] {
                return Err(format!(
                    "索引 {} 超出张量维度 {} 的范围（大小：{}）",
                    idx, dim, self.shape[dim]
                ));
            }
        }

        let index = self.calculate_index(&indices);
        self.data[index] = value;

        return Ok(());
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let total_size: usize = new_shape.iter().product();
        if total_size != self.numel().unwrap() {
            return Err(format!(
                "新形状 {:?} 的元素总数 {} 与原形状的元素总数 {} 不匹配",
                new_shape,
                total_size,
                self.numel().unwrap()
            ));
        }

        self.shape = new_shape;
        self.strides = Self::calculate_strides(&self.shape);

        return Ok(());
    }

    pub fn reshaped(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let total_size: usize = new_shape.iter().product();
        if total_size != self.numel().unwrap() {
            return Err(format!(
                "新形状 {:?} 的元素总数 {} 与原形状的元素总数 {} 不匹配",
                new_shape,
                total_size,
                self.numel().unwrap()
            ));
        }

        let new_data = self.data.clone();
        let new_strides = Self::calculate_strides(&new_shape);

        return Ok(Tensor {
            data: new_data,
            shape: new_shape,
            strides: new_strides,
        });
    }

    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        return strides;
    }

    fn calculate_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            index += idx * self.strides[i];
        }

        return index;
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f32;

    fn index(&self, indices: &[usize]) -> &Self::Output {
        return self.get(indices).unwrap();
    }
}

impl IndexMut<&[usize]> for Tensor {
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        return self.get_mut(indices).unwrap();
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor(shape: {:?}, data: [", self.shape)?;

        fn format_data(
            f: &mut fmt::Formatter,
            data: &[f32],
            shape: &[usize],
            strides: &[usize],
            offset: usize,
        ) -> fmt::Result {
            if shape.is_empty() {
                return write!(f, "{}", data[offset]);
            }

            let dim_size = shape[0];
            let inner_shape = &shape[1..];
            let inner_strides = &strides[1..];

            write!(f, "[")?;
            for i in 0..dim_size {
                let new_offset = offset + i * strides[0];
                if !inner_shape.is_empty() {
                    format_data(f, data, inner_shape, inner_strides, new_offset)?;
                } else {
                    write!(f, "{}", data[new_offset])?;
                }

                if i < dim_size - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")
        }

        format_data(f, &self.data, &self.shape, &self.strides, 0)?;
        write!(f, "])")
    }
}
