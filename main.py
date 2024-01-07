import numpy as np
import matplotlib.pyplot as plt
from A2_loadtxt_nptorch  import  loadtxt_np
import os

# https://blog.csdn.net/lhys666/article/details/120509520
# Matplotlib 绘制 双轴 图
#  还能改颜色
# 曲线种类
#https://www.zhihu.com/tardis/bd/art/139052035?source_id=1001

def GetData():
    xdata = np.arange(0.01, 10.0, 0.01) # t ∈ [ 0.01 , 10 ] t \in [0.01, 10]t∈[0.01,10] ，且以 0.01 作为 间距
    data1 = np.exp(xdata)
    data2 = np.sin(2 * np.pi * xdata)
    return xdata,data1,data2

#SetAxis() 函数用于设置图表的轴标签和字体大小，并返回了主轴 ax1，ax2和 fig 对象。
def SetAxis(colorlist_Axis):
    plt.title("")
    fig, ax1 = plt.subplots()  # ax1 即为主轴
    ax1.set_xlabel('Epoch',fontsize=15)  # set_xlabel 就是 设置 x 轴的标签
    ax1.set_ylabel('Accuracy', fontsize=15,color=colorlist_Axis[1])

    # plt.tick_params(labelsize=13)  # 刻度字体大小13
    ax1.tick_params(axis='x', labelsize=13,labelcolor=colorlist_Axis[0]) # tick_params 在这里的作用是 设置 y 轴的颜色
    ax1.tick_params(axis='y', labelsize=13,labelcolor=colorlist_Axis[1])

    #设置第二条轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    return ax1,fig,ax2

# supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
# dashes=(5, 1)
# # exitlist_loss,colorlist_mAP50,titile_loss,titile_mAP50,colorlist_loss,colorlist_mAP50,fig
# def Draw(xdata,data1,data2,ax1,ax2,fig):
# plot 画折线图   # o原点标记 marker  markersize=100 linestyle  dashed线形状
# linestyle  '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'

#Draw() 函数根据输入的数据列表，绘制出损失值和准确率的变化曲线。它遍历损失值列表和准确率列表，用不同的颜色绘制曲线，并在图上显示图例。
def Draw(datalist_loss,datalist_mAP50,titile_loss,titile_mAP50,colorlist_loss,colorlist_mAP50,fig,ax1,ax2):
    ####### plot 线条 ###############
    #datalist_mAP50
    # linewidth =5
    for i in range(len(datalist_loss)):

        ax1.plot(datalist_loss[i][:,0], datalist_loss[i][:,1], color=colorlist_loss[i],
                 linestyle='solid',label=titile_loss[i],linewidth =2.5)  # plot 画折线图   # o原点标记 marker marker='D'  markersize=100 linestyle  dashed线形状
        ax2.plot(datalist_mAP50[i][:,0], datalist_mAP50[i][:,1], color=colorlist_mAP50[i+1],
                 linestyle='dashed', label=titile_mAP50[i], linewidth=2.5)
    fig.legend(fontsize=13, bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)
    plt.show()

#这个函数是为了从一个较大的数据集中提取固定间隔的数据点，并将其重新整理成一个包含两列数据（第一列为索引，第二列为提取的数据点）的NumPy数组。
def ExtractData_loss(np_content,needsum=7):
    intervalsum = int (np_content.shape[0]/needsum)
    import numpy as np  # 导入NumPy模块
    npArray = np.zeros((needsum, 2))
    for i in range(needsum):
        npArray[i][0]=i
        npArray[i][1] = np_content[i][1]
    return npArray


# c cyan b蓝色  g绿色  r红色 c青色 m品红色  k黑色  w白色  m——magenta
#  upper left    upper right    lower left   lower right    center left    center right
# pip install pyQt5 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

#检查文件是否为空，如果不为空，那么就清空
def check_and_clear_file(filename):
    # 定义文件夹路径
    directory = './line/'
    # 检查文件夹是否存在，不存在则创建文件夹
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 然后在文件夹中创建文件（如果不存在）
    filename = './line/p_trainacc111.txt'
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            pass  # 这里什么也不做，只是创建一个空的文件

    # 检查文件内容是否为空
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        if content.strip():  # 如果文件内容不为空
            print("File is not empty. Clearing content...")
            with open(filename, 'w', encoding='utf-8') as f_clear:
                f_clear.truncate(0)  # 清空文件内容
            print("Content cleared.")
        else:
            print("File is empty.")


#写入文件
def WriteTxtLine(filename , context):
    with open(filename, 'a+',encoding='utf-8') as fp:
        fp.write('\n'+ context )
        fp.close()

if __name__ == '__main__':
    #写入数据
    acc_data = [0.72, 0.83, 0.86, 0.90, 0.92, 0.95, 0.98]      # 写入准确率数据到 p_trainacc.txt 文件中
    loss_data = [0.95, 0.92, 0.60, 0.58, 0.47, 0.46, 0.40]      # 写入损失数据到 p_trainloss.txt 文件中
    # 调用函数检查并清空文件内容（如果不为空）
    check_and_clear_file('./line/p_trainacc111.txt')
    check_and_clear_file('./line/p_trainloss111.txt')

    for epoch in range(len(acc_data)):
        WriteTxtLine('./line/p_trainacc111.txt',str(epoch)+ " " + str(acc_data[epoch]) )
    for epoch in range(len(loss_data)):
        WriteTxtLine('./line/p_trainloss111.txt',str(epoch)+ " " + str(loss_data[epoch]) )

# # AB线_新方法与原始对比；；；；读取文件，
    datalist_mAP50 = []
    np_content = loadtxt_np('./line/p_trainacc111.txt');datalist_mAP50.append(np_content)
    # np_content = loadtxt_np('./line/p2_trainacc.txt');datalist_mAP50.append(np_content)
    # np_content = loadtxt_np('./line/p2_sn_trainacc.txt');datalist_mAP50.append(np_content)

    datalist_loss= []
    np_content = loadtxt_np('./line/p_trainloss111.txt');np_content = ExtractData_loss(np_content);datalist_loss.append(np_content)
    # np_content = loadtxt_np('./line/p2_trainloss.txt');np_content = ExtractData_loss(np_content); datalist_loss.append(np_content)
    # np_content = loadtxt_np('./line/p2_sn_trainloss.txt');np_content = ExtractData_loss(np_content);datalist_loss.append(np_content)

    titile_loss=['Loss_Point','Loss_Point2','Loss_Our']
    titile_mAP50 = ['Accuracy_Point','Accuracy_Point2','Accuracy_Our']
    colorlist_loss =['limegreen','deepskyblue','darkorange']      # deepskyblue 深澜  darkorange绿
    colorlist_mAP50 = ['limegreen','deepskyblue','darkorange']

    # ##### 以下尽量不要自定义
    colorlist_Axis=['k','k']
    ax1,fig,ax2  = SetAxis(colorlist_Axis)
    #画图
    Draw(datalist_mAP50,datalist_loss,titile_mAP50,titile_loss,colorlist_loss,colorlist_mAP50,fig,ax1,ax2)
    #写一个函数用来检测文件内容是否为空，如果不为空，那么清空内容