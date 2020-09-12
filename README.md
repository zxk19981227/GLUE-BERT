# GLUE-BERT
This is a implemmention of Bert's expertiments on GLUE dataset
首先使用pip install requirements.txt
使用方法:
#CoLA
python CoLA.py就可以完成训练，我设置的epoch是10，但是出现了轻微过拟合
#SST
python SST-2.py 完成训练，5e-5的条件下出现过拟合，最佳测试集合