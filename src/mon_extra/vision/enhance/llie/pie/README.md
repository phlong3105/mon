# PIE

This code implements the paper of **"A Probabilistic Method for Image Enhancement With Simultaneous Illumination and Reflectance Estimation"** ,TIP,in 2015. Since the original project,[Xueyang Fu](http://smartdsp.xmu.edu.cn/Appendix.html), is wirten by **Matlab** and provide **".p"** file for encryption,this project is made for throughly understand the idea behind the paper.

## Usage

Run the command "python main.py -i dataDir", where "dataDir" specifies the figures to be processed. For more information, please see the "main.py" in detail.


## Sample Results

input|illumination|reflectance|enhanced
---- |-----|------|-------
<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/1.bmp" width = "200" height = "300" alt="in"/>|<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/1_I.jpg" width = "200" height = "300" alt="il"/>|<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/1_R.jpg" width = "200" height = "300" alt="ref"/>|<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/1_res.jpg" width = "200" height = "300" alt="res"/>


input|enhanced
----|-----
<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/3.bmp" width = "400" height = "400" alt="3in"/>|<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/3_res.jpg" width = "400" height = "400" alt="3out"/>
<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/4.bmp" width = "400" height = "400" alt="4in"/>|<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/4_res.jpg" width = "400" height = "400" alt="4out"/>
<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/6.bmp" width = "400" height = "400" alt="6in"/>|<img src="https://github.com/DavidQiuChao/PIE/blob/main/figs/6_res.jpg" width = "400" height = "400" alt="6out"/>


