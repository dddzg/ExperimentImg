
// ExperimentImgDlg.h : 头文件
//

#pragma once
#include "afxwin.h"
#include "afxcmn.h"
#include <string>
#include <cstring>
#include "ImageProcesser.h"
#include <ctime>
using namespace std;
// CExperimentImgDlg 对话框
class CExperimentImgDlg : public CDialogEx
{
// 构造
public:
	CExperimentImgDlg(CWnd* pParent = NULL);	// 标准构造函数
	void LogInfo(const string& str);
	void paintPic(CImage* img, CStatic& pic);
// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_EXPERIMENTIMG_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持
	CImage* getImage() { return m_pImgSrc; }
// 实现
protected:
	HICON m_hIcon;
	CImage * m_pImgSrc;
	CImage * m_pImgDst;
	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonOpen();
	CEdit mEditInfo;
	CStatic picLeft;
	CStatic picRight;
	CStatic messageText;
	CStatic labelText;
	CSliderCtrl slider;
	CStatic threadNum;
	afx_msg void OnNMCustomdrawSlider1(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnBnClickedButton2();
	CButton checkBox;
	CComboBox comboBox;
	afx_msg void OnBnClickedButton3();
};
