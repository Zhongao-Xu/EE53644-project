# Nsight tutorial

1. Install Nsight  
https://developer.nvidia.com/nsight-systems/get-started

2. Add Installed Path to Path Variable (System)  
For example, my path is "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.1\target-windows-x64"

3. Run nsight profile in terminal  
`nsys profile python .\VGG-16_test.py`

4. View Report in GUI  
Just double click the `.nsys-rep` file, it will automatically view the report. Details can be viewed by `Ctrl + Scroll up`.