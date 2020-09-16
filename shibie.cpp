#include<cv.h>;
#include<cxcore.h>;
#include<highgui.h>;
#include <vector>
using namespace cv;
using namespace std;

//1、预处理得到二值图像；2、提取车牌区域；3、提取车牌中字符；4、识别字符

IplImage* img;
IplImage* img_1;
IplImage* img_2;
IplImage* img4;
vector<CvRect>rects;
CvRect rection;


void pip(object_image)
{
	
	IplImage* img;
	IplImage* dst;
	IplImage* dst1;
	dst=cvCreateImage(cvGetSize(object_image), object_image->depth,object_image->nChannels);
	dst1=cvCreateImage(cvGetSize(object_image), object_image->depth,object_image->nChannels);
	for(int j=0;j<=9;j++)
		for(int i=1;i<=9;i++)
		{
			double a1,a2[10],a3[10],a[10];
			char name[1]={0};
			char bu_name[1]={0};
			sprintf(name,"muban/%d.bmp",i-1); 
			sprintf(bu_name,"muban/%d.bmp",i); 
			// 加载模板图像
			img=cvLoadImage(name);
			dst=cvLoadImage(bu_name);
			// 计算点积
			a1=cvDotProduct(img1,img1);
			a3[i-1]=cvDotProduct(img,img);
			a2[i-1]=cvDotProduct(img1,img);	
			a[i-1]=a1+a3[i-1]-2*a2[i-1];
			// 比较相似度
			if(a[i-1]>=a[i])
			{
				cvCopy(img,dst1);
				cvSaveImage(bu_name,img);
				cvSaveImage(name,dst);
			}
			cvShowImage("muban",dst1);
			cvWaitKey(100);
		}
	// 释放内存
	cvReleaseImage(&img);
	cvReleaseImage(&dst);
	cvReleaseImage(&dst1);
}



int main()
{
	// 创建窗口+加载图像
	namedWindow("chepai",1);
	img=cvLoadImage("che.jpg",1);
	// 创建图像
	img_1=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	//创建四个单通道图像
	IplImage* Bimg=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);//1为有1个通道
	IplImage* Gimg=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	IplImage* Rimg=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	IplImage* H=cvCreateImage(cvGetSize(img_1),IPL_DEPTH_8U,1);
	//将原图像分离到对应的三通道中
	cvSplit(img,Bimg,Gimg,Rimg,0);
	int height=img->height;
	int width=img->width;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			// 创建四个CvScalar，来存储三个图像通道的像素值
			CvScalar Bs,Gs,Rs,Hs;
			Bs=cvGet2D(Bimg,i,j);
			Gs=cvGet2D(Gimg,i,j);
			Rs=cvGet2D(Rimg,i,j);
			Hs=cvGet2D(H,i,j);
			//对每个通道对应像素进行加权计算复制到对应单通道图像种
			Hs.val[0]=0.302*Rs.val[0]+0.558*Gs.val[0]+0.110*Bs.val[0];
			cvSet2D(H,i,j,Hs);
		}
	}
	
	// 对应得到灰度图像
	cvMerge(H,NULL,NULL,NULL,img_1);
    // 阈值滤波得到黑白图像， CV_THRESH_OTSU是自适应阈值
	cvThreshold(img_1,img_1,0,255,CV_THRESH_OTSU);	
	cvShowImage("chepai",img_1);
	
	// 复制到img2
	img_2=cvCloneImage(img_1);	
	
	// 平滑图像减少图像上的噪点
	cvSmooth(img_2,img_1,CV_BLUR,2,1);
	// Sobel滤波对图像进行边缘检测
	cvSobel(img_1,img_1,2,0,1);
	cvShowImage("chepai",img_1);
	// 创建核函数
    IplConvKernel* P;
	P=cvCreateStructuringElementEx( 4, 1, 3.90, CV_SHAPE_RECT, NULL );	
	IplConvKernel* B;
	B=cvCreateStructuringElementEx(2, 3, 0, 0, CV_SHAPE_RECT, NULL );	
	
	// cvDilate膨胀，使不连通的图像合并成块
	cvDilate(img_1,img_1,P,2);
	// cvErode腐蚀，消除较小独点如噪音
	cvErode(img_1,img_1,P,2);
	// 连续膨胀腐蚀得到较好效果
	cvDilate(img_1,img_1,P,2);
    cvErode(img_1,img_1,P,5);
	cvErode(img_1,img_1,B,5);
	cvDilate(img_1,img_1,NULL,28);
	cvErode(img_1,img_1,B,5);
	cvErode(img_1,img_1,P,5);
	cvDilate(img_1,img_1,P,80);
	cvDilate(img_1,img_1,B,20);
	
    CvMemStorage *storage=cvCreateMemStorage(0);
	CvSeq *contours=0;
	// 从二值图像中提取轮廓,返回值为轮廓的数目
	cvFindContours(img_1,storage,&contours);
	// 在原图上绘制轮廓
	cvDrawContours(img,contours,cvScalar(255,0,255),cvScalar(0,0,0),100);
	//找到轮廓最小方框
	CvRect area_Rect = cvBoundingRect(contours,0);  
	// 绘制轮廓最小方框
	cvRectangleR(img,area_Rect,CV_RGB(255,0,0));  
	cvShowImage("chepai",img);
	// 提取图像方框区域
	cvSetImageROI(img, area_Rect);
	IplImage *img2 = cvCreateImage(cvGetSize(img),
		img->depth,
		img->nChannels);
	cvCopy(img, img2, NULL);
	// 还原图像
	cvResetImageROI(img);
	cvSaveImage("save/roi.jpg",img2);
	IplImage*  img3=cvCreateImage(cvGetSize(img2),IPL_DEPTH_8U,1);	
	// 转化区域原图像为灰度图像
	cvCvtColor(img2,img3,CV_BGR2GRAY);	
	// 阈值滤波得到二值图
	cvThreshold(img3,img3,0,255,CV_THRESH_OTSU); 
	cvShowImage("chepai",img3);
	CvScalar s;
	int bottom=img3->height;
	int top=0;
	int left=0;
	int right=img3->width;
	// 从上到下，从左至右搜索字符
	for(int i=0;i<img3->height;i++)
	{
		for(int j=0;j<img3->width;j++)
		{
			s=cvGet2D(img3,i,j);
			// 字符存在标志，找到最顶部位置
			if(s.val[0]>50)
			{
				top=i;
				i=img3->height;
				break;
			}
		}
	}
	for(int i=img3->height-1;i>=0;i--)
	{
		for(int j=0;j<img3->width;j++)
		{
			s=cvGet2D(img3,i,j);
			// 字符存在标志，找到最左边位置
			if(s.val[0]>50)
			{
				left=i;
				i=-1;
				break;
			}
		}
	}
	// 开始提取每一个字符
	bool lab=FALSE;
	bool black =FALSE;
	bool change=FALSE;
	//int num =0;
	// 由于字符数目固定，设定为 7
	for(int k=0;k<7;k++)
	{
		for(int j=0;j<img3->width;j++)
		{
			int cout=0;
			for(int i=0;i<img3->height;i++)
			{	
				s=cvGet2D(img3,i,j);
				// 统计变化次数
				if((s.val[0]>50)&&(!change))
				{
					cout++;
					change=TRUE;
				}
				else if((s.val[0]<50)&&(change))
				{
					cout++;
					change=FALSE;
				}
			}
            // 如果cout超过4说明将进入下一个字符
			// 注意不同的图像分辨率不同，数目也会有所差异
			if((!lab)&&(cout>4))
			{
				left=j-3;
				lab=TRUE;
			}
			// 到达图像区域边缘
			if(j==img3->width-1)
				break;
			// 如果lab为true，cout小于最大数目，且字符形态正确则裁剪相应区域
			if(lab&&(cout<8)&&(j>(left+8)))
			{
				right=j+2;
				lab=FALSE;
				CvPoint pt1;
				pt1.x=left;
				pt1.y=top;
				rection.x=pt1.x+1;
				rection.y=pt1.y;
				rection.width=right-left+1;
				rection.height=bottom-top;
				cvSetImageROI(img3,rection);
				img4 = cvCreateImage(cvGetSize(img3),
							   img3->depth,
							img3->nChannels);
				cvCopy(img3, img4);

				cvResetImageROI(img3);
				cvShowImage("chepai",img4);
				// 等待
				waitKey(1000);
				// 保存字符图像
				while(1!=0)
				{
					char c[1]={0};
					sprintf(c,"save/%02d.jpg",k++); 
					cvSaveImage(c,img4);
					break;
				}
                
				//字符识别
				pip(img4);
							  
			}
		}
	}
	// 释放内存
	cvWaitKey(0);
	cvDestroyWindow("chepai");
	cvReleaseImage(&img);
	cvReleaseImage(&img_1);
	return 0;
}
