
// ExperimentImgDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "ExperimentImg.h"
#include "ExperimentImgDlg.h"
#include "afxdialogex.h"
using namespace std;
#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CExperimentImgDlg 对话框



CExperimentImgDlg::CExperimentImgDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_EXPERIMENTIMG_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	//加载对话框的时候初始化
	m_pImgSrc = NULL;
}

void CExperimentImgDlg::LogInfo(const string & str)
{
	this->messageText.SetWindowTextW(CString(str.c_str()));
}


void CExperimentImgDlg::paintPic(CImage * img, CStatic & pic)
{
	int height;
	int width;
	CRect rect;
	CRect rect1;
	height = img->GetHeight();
	width = img->GetWidth();
	pic.GetClientRect(&rect);
	CDC *pDC = pic.GetDC();
	SetStretchBltMode(pDC->m_hDC, STRETCH_HALFTONE);
	if (width <= rect.Width() && height <= rect.Width())
	{
		rect1 = CRect(rect.TopLeft(), CSize(width, height));
		img->StretchBlt(pDC->m_hDC, rect1, SRCCOPY);
	}
	else
	{
		float xScale = (float)rect.Width() / (float)width;
		float yScale = (float)rect.Height() / (float)height;
		float ScaleIndex = (xScale <= yScale ? xScale : yScale);
		rect1 = CRect(rect.TopLeft(), CSize((int)width*ScaleIndex, (int)height*ScaleIndex));
		img->StretchBlt(pDC->m_hDC, rect1, SRCCOPY);
	}
	ReleaseDC(pDC);
}


void CExperimentImgDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT_INFO, mEditInfo);
	DDX_Control(pDX, IDC_PICTURE, picLeft);
	DDX_Control(pDX, IDC_PICTURE2, picRight);
	DDX_Control(pDX, IDC_STATIC3, messageText);
	DDX_Control(pDX, IDC_STATIC2, labelText);
	DDX_Control(pDX, IDC_SLIDER1, slider);
	DDX_Control(pDX, IDC_STATIC5, threadNum);
	DDX_Control(pDX, IDC_CHECK1, checkBox);
	DDX_Control(pDX, IDC_COMBO1, comboBox);
}

BEGIN_MESSAGE_MAP(CExperimentImgDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPEN, &CExperimentImgDlg::OnBnClickedButtonOpen)
//	ON_EN_CHANGE(IDC_EDIT1, &CExperimentImgDlg::OnEnChangeEdit1)
//	ON_EN_CHANGE(IDC_EDIT_INFO, &CExperimentImgDlg::OnEnChangeEditInfo)
ON_NOTIFY(NM_CUSTOMDRAW, IDC_SLIDER1, &CExperimentImgDlg::OnNMCustomdrawSlider1)
ON_BN_CLICKED(IDC_BUTTON2, &CExperimentImgDlg::OnBnClickedButton2)
ON_BN_CLICKED(IDC_BUTTON3, &CExperimentImgDlg::OnBnClickedButton3)
END_MESSAGE_MAP()


// CExperimentImgDlg 消息处理程序

BOOL CExperimentImgDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);
	
	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	mEditInfo.SetWindowTextW(CString("File Path"));
	this->labelText.SetWindowTextW(CString("线程数"));
	this->slider.SetRange(1, 10);
	this->slider.SetPos(1);
	this->threadNum.SetWindowTextW(CString(to_string(this->slider.GetPos()).c_str()));
	this->comboBox.AddString(CString("椒盐噪声"));
	this->comboBox.AddString(CString("中值滤波"));
	this->comboBox.AddString(CString("双三阶插值（缩放）"));
	this->comboBox.AddString(CString("双三阶插值（旋转）"));
	this->comboBox.AddString(CString("自动色阶"));
	this->comboBox.AddString(CString("自动白平衡"));
	this->comboBox.AddString(CString("自适应双边滤波"));
	this->comboBox.AddString(CString("融合左右图"));
	this->comboBox.SetCurSel(0);
	srand(time(nullptr));
	/************************************************************************/
	/* 设置console															*/
	/************************************************************************/
	AllocConsole();
	freopen("CONOUT$", "w+t", stdout);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CExperimentImgDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CExperimentImgDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
		if (m_pImgSrc != NULL)
		{
			this->paintPic(m_pImgSrc, picLeft);
		}
		if (m_pImgDst != NULL)
		{
			this->paintPic(m_pImgDst, picRight);
		}
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CExperimentImgDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CExperimentImgDlg::OnBnClickedButtonOpen()
{
	// TODO: 在此添加控件通知处理程序代码
	TCHAR szFilter[] = _T("JPEG(*jpg)|*.jpg|*.bmp|TIFF(*.tif)|*.tif|All Files （*.*）|*.*||");
	CString filePath("");
	
	CFileDialog fileOpenDialog(TRUE, NULL, NULL, OFN_HIDEREADONLY, szFilter);
	if (fileOpenDialog.DoModal() == IDOK)
	{
		VERIFY(filePath = fileOpenDialog.GetPathName());
		CString strFilePath(filePath);
		mEditInfo.SetWindowTextW(strFilePath);	//在文本框中显示图像路径

		if (m_pImgSrc != NULL)
		{
			m_pImgSrc->Destroy();
			delete m_pImgSrc;
		}
		m_pImgSrc = new CImage();
		m_pImgSrc->Load(strFilePath);

		this->Invalidate();

	}
}



void CExperimentImgDlg::OnNMCustomdrawSlider1(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMCUSTOMDRAW pNMCD = reinterpret_cast<LPNMCUSTOMDRAW>(pNMHDR);
	this->threadNum.SetWindowTextW(CString(to_string(this->slider.GetPos()).c_str()));
	// TODO: 在此添加控件通知处理程序代码
	*pResult = 0;
}


void CExperimentImgDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	int threadNum = this->slider.GetPos();
	bool useGPU = this->checkBox.GetCheck();
	CString typeName;
	auto index = this->comboBox.GetCurSel();
	this->comboBox.GetLBText(index, typeName);
	if (this->m_pImgSrc) {
		this->LogInfo("处理中");
		auto initTime = clock();
		if (typeName == "融合左右图") {
			this->m_pImgDst=ImageProcesser::merge(m_pImgSrc, m_pImgDst, double(threadNum)/10,useGPU);
		}else {
			ImageProcesser imgPro(this->m_pImgSrc, typeName, threadNum, useGPU);
			this->m_pImgDst = imgPro.go();
		}
		this->paintPic(m_pImgDst, picRight);
		this->LogInfo(string("处理完成。耗时")+to_string(clock()-initTime)+string("毫秒"));
	}
	else {
		this->LogInfo("请先打开图片，加载一张图片进来！");
	}
}

void CExperimentImgDlg::OnBnClickedButton3()
{
	if (this->m_pImgDst){
		delete this->m_pImgSrc;
		this->m_pImgSrc = this->m_pImgDst;
		this->paintPic(this->m_pImgSrc, this->picLeft);
		this->m_pImgDst = nullptr;
	}
	else {
		this->LogInfo("右边没有图片");
	}
	// TODO: 在此添加控件通知处理程序代码
}
