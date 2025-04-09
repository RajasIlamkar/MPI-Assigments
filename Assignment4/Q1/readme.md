# CUDA Sum of First N Integers (Different Tasks Per Thread)

This CUDA program demonstrates how different threads in a CUDA kernel can perform **distinct tasks**.

## 🔧 What the Code Does

The code calculates the **sum of the first N integers (N = 1024)** using two different methods, with each method assigned to a separate CUDA thread:

- **Thread 0:** Uses an **iterative approach** (loop-based sum)
- **Thread 1:** Uses a **mathematical formula** (sum = n(n - 1) / 2)

This helps illustrate how CUDA threads can be assigned different responsibilities within the same kernel.

## 🚀 How It Works

1. Define `N = 1024`
2. Create and fill an array with values from `0` to `1023`
3. Allocate memory on the GPU for input and output
4. Launch a CUDA kernel with at least **2 threads**
5. Thread 0 performs the iterative sum
6. Thread 1 uses the direct formula
7. Copy the results back to the host and print them

## 🧪 Output Example

