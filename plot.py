 
 
import matplotlib.pyplot as plt

 
def tensorboard_smoothing(x,smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x
 
if __name__ == '__main__':
    
    fig, ax1 = plt.subplots(1, 1)    # a figure with a 1x1 grid of Axes
    
    #设置上方和右方无框
    ax1.spines['top'].set_visible(False)                   # 不显示图表框的上边框
    ax1.spines['right'].set_visible(False)  
 
    
    ckpt = [0,1,2,3,4,5,6]
    success = [0, 0.82, 0.77, 0.93, 0.85, 0.76,0.86]
    len = [267.53,114,102.41,95,81.82,96.58,96.35]



    ax1.plot(ckpt, success, color="r",label='pick success rate')
    #ax1.plot(ckpt, len, color="b",label='pick step number')
    #ax1.plot(len_mean3['Step'], tensorboard_smoothing(len_mean3['Value'], smooth=0.6), color="g",label='model-free ')
    
   

 
    plt.legend(loc = 'best')
 
    ax1.grid()
    ax1.set_xlabel("ckpt number")
    ax1.set_ylabel("success rate")
   # ax1.set_xlim([0,2.5e6])
    ax1.set_title("Running depth_pick test")
    plt.show()

    fig.savefig(fname='./result'+'.png', format='png')