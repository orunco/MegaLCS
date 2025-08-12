# MegaLCS

**An OpenCL LCS(Longest Common Subsequence) parallel MinMax algorithm (~150 lines) supporting MILLION elements on a single GPU, ULTRA-FAST performance, and controllable time and memory usage.**

Keywords: Longest Common Subsequence, LCS, CUDA, OpenCL, Parallel Computing, MinMax

---

For example, based on execution results from a Tesla P40 environment, comparing **two integer arrays of length 1 million** takes approximately **only 9.5 seconds**.

```
| Runtime       | MAX     | STEP | Mean         | Allocated   |
|-------------- |-------- |----- |-------------:|------------:|
| .NET 8.0      | 65536   | 256  |     138.3 ms |   579.05 KB |
| NativeAOT 8.0 | 65536   | 256  |     137.8 ms |   578.98 KB |
| .NET 8.0      | 1048576 | 256  |   9,532.7 ms |  8739.59 KB |
| NativeAOT 8.0 | 1048576 | 256  |   9,533.7 ms |  8739.51 KB |
| .NET 8.0      | 2097152 | 256  |  36,642.7 ms | 17443.59 KB |
| NativeAOT 8.0 | 2097152 | 256  |  36,642.4 ms | 17443.51 KB |
| NativeAOT 8.0 | 4194304 | 256  | 142,976.1 ms | 34851.51 KB |
```

![Preview](./Preview-device.png)

## Why?

Although the Longest Common Subsequence problem has been extensively studied, most classic algorithms have a time complexity of O(n²). While various optimized variants exist, such as Myers(The Myers algorithm is very good, but it has a small drawback: it cannot be parallelized.), they often come with assumptions or constraints that limit their ability to handle large-scale modifications. In such cases, only the basic LCS algorithm can be applied. However, computing LCS for arrays of length 1 million on CPU is practically infeasible — GPU acceleration becomes essential.

There has been extensive research in this area, but existing implementations tend to be complex, and more importantly, most related papers lack publicly available source code :(. 

My goal was to solve this real-world problem with a simple and practical implementation. After repeated derivations, I discovered (perhaps reinvented) and implemented a concise and fully parallel solution. The key insight lies in computing `LeftTopWeight` with data-independent operations:

**int leftTopWeight = Math.Min(leftWeight, topWeight);**

A prototype implementation can be found in `CpuLCS_MinMax()`. For broader compatibility (e.g., if you have a powerful CPU and OpenCL support), the actual implementation uses OpenCL, with the kernel function named `KernelLCS_MinMax` (**the actual code is less than 100 lines**), and the host function named `HostLCS_WaveFront` (**core code is less than 50 lines**). Converting it to CUDA would also be straightforward.

### Proof

In the dynamic programming (DP) weight matrix of LCS, for the cell `dp[i][j]`, the value at its top-left corner `lefttop` (i.e., `dp[i-1][j-1]`) indeed equals the minimum of the left value `left` (`dp[i][j-1]`) and the top value `top` (`dp[i-1][j]`). Below is the proof process:

##### 1. Proof Steps

1. **Non-decreasing Property of the DP Table**  
   In the DP table for LCS, the value of each cell satisfies the non-decreasing property: when moving right or down, the value does not decrease. That is:
   - `dp[i][j] ≥ dp[i-1][j]` (value does not decrease when moving down)
   - `dp[i][j] ≥ dp[i][j-1]` (value does not decrease when moving right)

2. **Analysis of the Recurrence Relation**  
   - If the current characters match (`X[i] == Y[j]`), then `dp[i][j] = dp[i-1][j-1] + 1`.
   - If the characters do not match, then `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.  
   From this, it follows that, regardless of whether the characters match, the value of `dp[i][j]` is always no less than the value of its top-left cell `dp[i-1][j-1]`.

3. **Deriving the Relationship Between `lefttop`, `left`, and `top`**  
   - Due to the non-decreasing property, the left cell `dp[i][j-1] ≥ dp[i-1][j-1]`.
   - Similarly, the top cell `dp[i-1][j] ≥ dp[i-1][j-1]`.  
   Therefore, `min(dp[i][j-1], dp[i-1][j]) ≥ dp[i-1][j-1]`.

4. **Proof by Contradiction**  
   Assume there exists some `i,j` such that `min(dp[i][j-1], dp[i-1][j}) > dp[i-1][j-1}`. This implies that both `dp[i][j-1]` and `dp[i-1][j]` are strictly greater than `dp[i-1][j-1}`. However, according to the recurrence rules:
   - The value of `dp[i][j-1]` comes from `max(dp[i-1][j-1}, dp[i][j-2})` or, in the case of a match, `dp[i-1][j-2}+1`. In any case, `dp[i][j-1]` is at least equal to `dp[i-1][j-1}`.
   - Similarly, the value of `dp[i-1][j]` is also at least equal to `dp[i-1][j-1}`.  
   If both are strictly greater than `dp[i-1][j-1}`, it contradicts the non-decreasing property of the DP table, as `dp[i-1][j-1]` should have been updated to a larger value during the recurrence process.

5. **Inductive Summary**  
   Through mathematical induction and concrete examples, it can be confirmed that in the DP table for LCS, `lefttop` always equals the minimum of `left` and `top`. This property is guaranteed by the non-decreasing nature of the DP table and the recurrence rules.

##### 2. Conclusion

For the DP weight matrix of the LCS algorithm, the top-left value `lefttop` of cell `dp[i][j]` satisfies:  
**lefttop = min(left, top)**  
where `left` is the left value (`dp[i][j-1]`) and `top` is the top value (`dp[i-1][j]`). This conclusion is rigorously proven by the non-decreasing property of the DP table and the recurrence rules, and its correctness is verified through examples.

Based on our comprehensive analysis, here is the English translation of the conclusion:

### Time and Space Complexity Analysis

#### Effective Parallelism
```
P = min(
    P_compute,     // Compute parallelism capacity
    P_memory,      // Memory bandwidth parallelism
    P_wavefront    // Wavefront parallelism limitation
)
```

Where:
- **P_compute**: Maximum concurrent threads supported by GPU compute units
- **P_memory**: Effective parallelism limited by memory bandwidth
- **P_wavefront**: Algorithm-inherent parallelism = min(M, N) per wavefront

#### Complexity Results

##### **Time Complexity:**

```
O((M+N) × max(1, (M+N)/P))
```

##### **Space Complexity:**

```
O(M+N)
```

##### Performance Scenarios

1. **GPU-abundant case**: When `P ≥ (M+N)`
   - Time complexity reduces to **O(M+N)**
   - Achieves optimal parallel efficiency

2. **GPU-limited case**: When `P < (M+N)`
   - Time complexity becomes **O((M+N)²/P)**
   - Performance degrades linearly with available parallelism

3. **Sequential case**: When `P = 1`
   - Time complexity degrades to **O(M×N)**
   - Equivalent to traditional DP approach

This GPU-accelerated LCS algorithm with wavefront parallelization achieves significant speedup over the classical O(M×N) approach, with actual performance bounded by the effective GPU parallelism capacity.





## Getting Started

### cpp
```bash
git clone https://github.com/orunco/MegaLCS.git
cd cpp\MegaLCSTest
cmake and run
```

Alternatively, you can import it into your own project and directly use the function interface:

```csharp
Mega::MegaLCSLen(const vector<int>& baseVals, const vector<int>& latestVals)
```


### csharp

```bash
git clone https://github.com/orunco/MegaLCS.git
cd csharp\MegaLCSTest
dotnet run -c Release
```

Alternatively, you can import it into your own project and directly use the function interface:

```csharp
Mega.MegaLCSLen(int[] baseVals, int[] latestVals)
```

That’s all.



## Requirements

The project currently uses C# as the primary development language for ease of development and debugging.

- .NET 8+
- Silk.NET.OpenCL 2.22.0 (MIT LICENSE)
- A powerful  OpenCL-compatible(GPU or Powerful CPU) device

## TODO

- Currently, only the length of the LCS is computed. Backtracking to retrieve one or more actual common subsequences is not yet implemented. Given that weights are already calculated, implementing backtracking should be feasible and may be added in the future. Theoretical derivation result: Memory and time are approximately twice the current results.

## License

Copyright (C) 2025 Pete Zhang, rivxer@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
