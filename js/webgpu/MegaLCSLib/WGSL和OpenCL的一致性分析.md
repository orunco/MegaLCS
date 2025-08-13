WebGPU 的 WGSL（WebGPU Shading Language）确实在设计上借鉴了 OpenCL 的许多概念，特别是在计算着色器（Compute Shader）方面。你提到的这些内置变量与 OpenCL 的内置变量非常相似，因为它们都遵循相同的并行计算模型。下面是详细对比：

---

### **1. 内置变量对比表**
| **WGSL (WebGPU)**               | **OpenCL**                   | **作用** |
|----------------------------------|-----------------------------|----------|
| `@builtin(global_invocation_id)` | `get_global_id(dim)`        | 当前工作项在全局NDRange中的坐标 |
| `@builtin(local_invocation_id)`  | `get_local_id(dim)`         | 当前工作项在工作组内的局部坐标 |
| `@builtin(workgroup_id)`         | `get_group_id(dim)`         | 当前工作组在全局NDRange中的坐标 |
| `@workgroup_size(X,Y,Z)`         | `__attribute__((reqd_work_group_size(X,Y,Z)))` | 定义工作组大小 |

---

### **2. 相似性分析**
#### **(1) 执行模型一致**
两者都基于 **分层并行模型**：
- **全局范围 (NDRange)**：所有并行工作项的集合
- **工作组 (Workgroup)**：局部同步的单位（类似 OpenCL 的 `local_work_size`）
- **工作项 (Invocation)**：最小的执行单元（类似 OpenCL 的 kernel 实例）

#### **(2) 索引计算方式相同**
例如，计算全局线性索引的公式在两种语言中几乎一致：
```wgsl
let global_index = global_id.x + global_id.y * global_size.x + global_id.z * global_size.x * global_size.y;
```
对应 OpenCL：
```c
size_t global_index = get_global_id(0) + get_global_id(1) * get_global_size(0) + get_global_id(2) * get_global_size(0) * get_global_size(1);
```

#### **(3) 内存模型类似**
| 概念               | WGSL                          | OpenCL                     |
|--------------------|-------------------------------|----------------------------|
| 私有内存           | 局部变量                      | `private` 内存            |
| 工作组共享内存      | `var<workgroup>` 变量         | `__local` 内存            |
| 全局内存           | `var<storage>` 缓冲区         | `__global` 缓冲区         |

---

### **3. 关键差异**
尽管概念相似，但存在一些重要区别：

| **特性**         | **WGSL**                     | **OpenCL**                |
|------------------|------------------------------|---------------------------|
| **语言基础**      | 基于 Rust 语法                | 基于 C99 语法             |
| **指针**          | 不支持裸指针                  | 支持指针运算              |
| **同步**          | `workgroupBarrier()`          | `barrier(CLK_LOCAL_MEM_FENCE)` |
| **动态工作组大小**| 必须编译时确定 `@workgroup_size` | 可运行时指定              |

---

### **4. 代码示例对比**
#### **WGSL 计算着色器**
```rust
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        output[idx] = input[idx] * 2;
    }
}
```

#### **OpenCL Kernel**
```c
__kernel void double_values(
    __global const int* input,
    __global int* output
) {
    size_t idx = get_global_id(0);
    if (idx < get_global_size(0)) {
        output[idx] = input[idx] * 2;
    }
}
```

---

### **5. 设计意图**
WebGPU 选择这种设计是为了：
1. **降低学习成本**：让熟悉 OpenCL/CUDA 的开发者快速上手
2. **保证安全性**：通过限制指针等特性避免 GPU 安全漏洞
3. **跨平台一致性**：抽象不同后端（Metal/Vulkan/D3D12）的差异

---

### **结论**
WGSL 的计算着色器部分**确实刻意模仿了 OpenCL 的设计**，但加入了更适合 Web 平台的限制和语法糖。如果你已经熟悉 OpenCL，转换到 WGSL 会非常自然，只需注意：
1. 语法从 C 风格变为 Rust 风格
2. 某些高级特性（如指针）被移除
3. 更严格的编译时检查

这种相似性使得 GPU 计算代码可以相对容易地在 OpenCL 和 WebGPU 之间迁移。