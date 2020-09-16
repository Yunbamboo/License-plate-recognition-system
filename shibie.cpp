#include<cv.h>;
#include<cxcore.h>;
#include<highgui.h>;
#include <vector>
using namespace cv;
using namespace std;

//1��Ԥ����õ���ֵͼ��2����ȡ��������3����ȡ�������ַ���4��ʶ���ַ�

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
			// ����ģ��ͼ��
			img=cvLoadImage(name);
			dst=cvLoadImage(bu_name);
			// ������
			a1=cvDotProduct(img1,img1);
			a3[i-1]=cvDotProduct(img,img);
			a2[i-1]=cvDotProduct(img1,img);	
			a[i-1]=a1+a3[i-1]-2*a2[i-1];
			// �Ƚ����ƶ�
			if(a[i-1]>=a[i])
			{
				cvCopy(img,dst1);
				cvSaveImage(bu_name,img);
				cvSaveImage(name,dst);
			}
			cvShowImage("muban",dst1);
			cvWaitKey(100);
		}
	// �ͷ��ڴ�
	cvReleaseImage(&img);
	cvReleaseImage(&dst);
	cvReleaseImage(&dst1);
}



int main()
{
	// ��������+����ͼ��
	namedWindow("chepai",1);
	img=cvLoadImage("che.jpg",1);
	// ����ͼ��
	img_1=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	//�����ĸ���ͨ��ͼ��
	IplImage* Bimg=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);//1Ϊ��1��ͨ��
	IplImage* Gimg=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	IplImage* Rimg=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	IplImage* H=cvCreateImage(cvGetSize(img_1),IPL_DEPTH_8U,1);
	//��ԭͼ����뵽��Ӧ����ͨ����
	cvSplit(img,Bimg,Gimg,Rimg,0);
	int height=img->height;
	int width=img->width;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			// �����ĸ�CvScalar�����洢����ͼ��ͨ��������ֵ
			CvScalar Bs,Gs,Rs,Hs;
			Bs=cvGet2D(Bimg,i,j);
			Gs=cvGet2D(Gimg,i,j);
			Rs=cvGet2D(Rimg,i,j);
			Hs=cvGet2D(H,i,j);
			//��ÿ��ͨ����Ӧ���ؽ��м�Ȩ���㸴�Ƶ���Ӧ��ͨ��ͼ����
			Hs.val[0]=0.302*Rs.val[0]+0.558*Gs.val[0]+0.110*Bs.val[0];
			cvSet2D(H,i,j,Hs);
		}
	}
	
	// ��Ӧ�õ��Ҷ�ͼ��
	cvMerge(H,NULL,NULL,NULL,img_1);
    // ��ֵ�˲��õ��ڰ�ͼ�� CV_THRESH_OTSU������Ӧ��ֵ
	cvThreshold(img_1,img_1,0,255,CV_THRESH_OTSU);	
	cvShowImage("chepai",img_1);
	
	// ���Ƶ�img2
	img_2=cvCloneImage(img_1);	
	
	// ƽ��ͼ�����ͼ���ϵ����
	cvSmooth(img_2,img_1,CV_BLUR,2,1);
	// Sobel�˲���ͼ����б�Ե���
	cvSobel(img_1,img_1,2,0,1);
	cvShowImage("chepai",img_1);
	// �����˺���
    IplConvKernel* P;
	P=cvCreateStructuringElementEx( 4, 1, 3.90, CV_SHAPE_RECT, NULL );	
	IplConvKernel* B;
	B=cvCreateStructuringElementEx(2, 3, 0, 0, CV_SHAPE_RECT, NULL );	
	
	// cvDilate���ͣ�ʹ����ͨ��ͼ��ϲ��ɿ�
	cvDilate(img_1,img_1,P,2);
	// cvErode��ʴ��������С����������
	cvErode(img_1,img_1,P,2);
	// �������͸�ʴ�õ��Ϻ�Ч��
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
	// �Ӷ�ֵͼ������ȡ����,����ֵΪ��������Ŀ
	cvFindContours(img_1,storage,&contours);
	// ��ԭͼ�ϻ�������
	cvDrawContours(img,contours,cvScalar(255,0,255),cvScalar(0,0,0),100);
	//�ҵ�������С����
	CvRect area_Rect = cvBoundingRect(contours,0);  
	// ����������С����
	cvRectangleR(img,area_Rect,CV_RGB(255,0,0));  
	cvShowImage("chepai",img);
	// ��ȡͼ�񷽿�����
	cvSetImageROI(img, area_Rect);
	IplImage *img2 = cvCreateImage(cvGetSize(img),
		img->depth,
		img->nChannels);
	cvCopy(img, img2, NULL);
	// ��ԭͼ��
	cvResetImageROI(img);
	cvSaveImage("save/roi.jpg",img2);
	IplImage*  img3=cvCreateImage(cvGetSize(img2),IPL_DEPTH_8U,1);	
	// ת������ԭͼ��Ϊ�Ҷ�ͼ��
	cvCvtColor(img2,img3,CV_BGR2GRAY);	
	// ��ֵ�˲��õ���ֵͼ
	cvThreshold(img3,img3,0,255,CV_THRESH_OTSU); 
	cvShowImage("chepai",img3);
	CvScalar s;
	int bottom=img3->height;
	int top=0;
	int left=0;
	int right=img3->width;
	// ���ϵ��£��������������ַ�
	for(int i=0;i<img3->height;i++)
	{
		for(int j=0;j<img3->width;j++)
		{
			s=cvGet2D(img3,i,j);
			// �ַ����ڱ�־���ҵ����λ��
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
			// �ַ����ڱ�־���ҵ������λ��
			if(s.val[0]>50)
			{
				left=i;
				i=-1;
				break;
			}
		}
	}
	// ��ʼ��ȡÿһ���ַ�
	bool lab=FALSE;
	bool black =FALSE;
	bool change=FALSE;
	//int num =0;
	// �����ַ���Ŀ�̶����趨Ϊ 7
	for(int k=0;k<7;k++)
	{
		for(int j=0;j<img3->width;j++)
		{
			int cout=0;
			for(int i=0;i<img3->height;i++)
			{	
				s=cvGet2D(img3,i,j);
				// ͳ�Ʊ仯����
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
            // ���cout����4˵����������һ���ַ�
			// ע�ⲻͬ��ͼ��ֱ��ʲ�ͬ����ĿҲ����������
			if((!lab)&&(cout>4))
			{
				left=j-3;
				lab=TRUE;
			}
			// ����ͼ�������Ե
			if(j==img3->width-1)
				break;
			// ���labΪtrue��coutС�������Ŀ�����ַ���̬��ȷ��ü���Ӧ����
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
				// �ȴ�
				waitKey(1000);
				// �����ַ�ͼ��
				while(1!=0)
				{
					char c[1]={0};
					sprintf(c,"save/%02d.jpg",k++); 
					cvSaveImage(c,img4);
					break;
				}
                
				//�ַ�ʶ��
				pip(img4);
							  
			}
		}
	}
	// �ͷ��ڴ�
	cvWaitKey(0);
	cvDestroyWindow("chepai");
	cvReleaseImage(&img);
	cvReleaseImage(&img_1);
	return 0;
}
