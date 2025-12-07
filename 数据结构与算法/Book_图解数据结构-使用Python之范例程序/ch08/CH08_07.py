import random

def inputarr(data,size):
    for i in range(size):
        data[i]=random.randint(1,100)
        
def showdata(data,size):
    for i in range(size):
        print('%3d' %data[i],end='')
    print()

def quick(d,size,lf,rg):
    #第一项键值为d[lf]
    if lf<rg:  #排序数列的左边与右边
        lf_idx=lf+1
        while d[lf_idx]<d[lf]:
            if lf_idx+1 >size:
                break
            lf_idx +=1
        rg_idx=rg
        while d[rg_idx] >d[lf]:
            rg_idx -=1
        while lf_idx<rg_idx:
            d[lf_idx],d[rg_idx]=d[rg_idx],d[lf_idx]
            lf_idx +=1
            while d[lf_idx]<d[lf]:
                lf_idx +=1
            rg_idx -=1
            while d[rg_idx] >d[lf]:
                rg_idx -=1
        d[lf],d[rg_idx]=d[rg_idx],d[lf]

        for i in range(size):
            print('%3d' %d[i],end='')
        print()
       
        quick(d,size,lf,rg_idx-1)   #以rg_idx为基准点分成左右两半以递归方式
        quick(d,size,rg_idx+1,rg)   #分别为左右两半进行排序直至完成排序               
		
def main():
    data=[0]*100
    size=int(input('请输入数组大小(100以下)：'))
    inputarr (data,size)
    print('您输入的原始数据是：')
    showdata (data,size)
    print('排序过程如下：')
    quick(data,size,0,size-1)
    print('最终的排序结果为：')
    showdata(data,size)

main()