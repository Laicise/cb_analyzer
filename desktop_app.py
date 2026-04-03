#!/usr/bin/env python3
"""
可转债价格预测工具 - GUI版本
基于机器学习模型预测可转债首日开盘价

运行方式:
    python3 desktop_app.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import warnings
warnings.filterwarnings('ignore')

from db.models import get_session, BondInfo, StockInfo
from analysis.ml_model_v6 import predict_price_v6, load_v6_model


class CBPredictorApp:
    """可转债预测工具主界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("可转债价格预测工具 v1.0")
        self.root.geometry("800x620")
        self.root.minsize(700, 500)
        
        self.center_window()
        self.model_loaded = False
        self.models = None
        self.metadata = None
        
        self.load_model_async()
        self.create_widgets()
    
    def center_window(self):
        self.root.update_idletasks()
        w, h = 800, 620
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f'{w}x{h}+{x}+{y}')
    
    def load_model_async(self):
        def load():
            try:
                self.models, self.metadata = load_v6_model()
                self.model_loaded = True
                self.root.after(0, lambda: self.status_label.config(
                    text=f"✅ 模型已就绪 (MAE: {self.metadata.get('mae_stack', 7.55):.2f}元)",
                    foreground="green"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(
                    text="❌ 模型加载失败", foreground="red"))

        threading.Thread(target=load, daemon=True).start()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        ttk.Label(main_frame, text="可转债首日开盘价预测",
                  font=("Arial", 20, "bold")).pack()
        ttk.Label(main_frame, text="基于机器学习模型的智能预测系统 | ML v6 Stacking",
                  font=("Arial", 10), foreground="gray").pack(pady=(0, 15))
        
        # 输入区域
        input_frame = ttk.LabelFrame(main_frame, text="输入可转债代码", padding="15")
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X)
        
        ttk.Label(row, text="债券代码:", font=("Arial", 12)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.code_entry = ttk.Entry(row, font=("Arial", 14), width=12)
        self.code_entry.pack(side=tk.LEFT, padx=(0, 15))
        self.code_entry.bind('<Return>', lambda e: self.predict())
        
        # 快捷按钮
        quick_frame = ttk.Frame(row)
        quick_frame.pack(side=tk.LEFT)
        ttk.Label(quick_frame, text="快捷:", font=("Arial", 9), foreground="gray").pack(side=tk.LEFT, padx=(0, 5))
        
        quick_codes = [("110074", "精达"), ("118067", "上26"), ("113701", "祥和"), ("127113", "长高")]
        for code, name in quick_codes:
            btn = ttk.Button(quick_frame, text=f"{code}", width=7,
                           command=lambda c=code: [self.code_entry.delete(0, tk.END), self.code_entry.insert(0, c)])
            btn.pack(side=tk.LEFT, padx=2)
        
        self.predict_btn = ttk.Button(input_frame, text="🔮 开始预测", 
                                      command=self.predict, style="Accent.TButton")
        self.predict_btn.pack(fill=tk.X, pady=(15, 0))
        
        # 结果区域
        result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding="15")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 带滚动条的文本
        text_frame = ttk.Frame(result_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scroll = ttk.Scrollbar(text_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(text_frame, font=("Consolas", 11), 
                                   wrap=tk.WORD, yscrollcommand=scroll.set,
                                   bg="#f5f5f5", relief=tk.FLAT)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.result_text.yview)
        
        # 状态栏
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="⏳ 加载模型...",
                                      font=("Arial", 9), foreground="orange")
        self.status_label.pack(side=tk.LEFT)
        
        ttk.Label(status_frame, text="v2.0 | ML v6 Stacking | 分位数回归",
                  font=("Arial", 9), foreground="gray").pack(side=tk.RIGHT)
        
        ttk.Style().configure("Accent.TButton", font=("Arial", 13, "bold"))
    
    def predict(self):
        code = self.code_entry.get().strip().zfill(6)
        
        if not code:
            messagebox.showwarning("提示", "请输入可转债代码")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("提示", "模型正在加载中，请稍候...")
            return
        
        self.predict_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏳ 预测中...", foreground="orange")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在分析，请稍候...\n")
        self.root.update()
        
        def do_predict():
            try:
                result = predict_price_v6(code)
                session = get_session()
                bond = session.query(BondInfo).filter_by(bond_code=code).first()
                session.close()
                self.root.after(0, lambda: self.show_result(result, bond, code))
            except Exception as e:
                self.root.after(0, lambda: self.show_error(str(e)))
        
        threading.Thread(target=do_predict, daemon=True).start()
    
    def show_result(self, result, bond, code):
        self.predict_btn.config(state=tk.NORMAL)
        self.status_label.config(text="✅ 预测完成", foreground="green")
        self.result_text.delete(1.0, tk.END)
        
        if result is None:
            self.result_text.insert(tk.END, "❌ 未找到该可转债\n\n")
            self.result_text.insert(tk.END, f"代码 {code} 不在数据库中\n")
            self.result_text.insert(tk.END, "请输入正确的6位可转债代码（如：110074）")
            return
        
        # 基本信息
        bond_name = bond.bond_name if bond else "N/A"
        stock_code = bond.stock_code if bond else "N/A"
        issue_size = f"{bond.issue_size:.2f}" if bond and bond.issue_size else "N/A"
        rating = bond.credit_rating if bond and bond.credit_rating else "N/A"
        listing_date = str(bond.listing_date)[:10] if bond and bond.listing_date else "N/A"
        cv = f"{bond.conversion_value:.1f}" if bond and bond.conversion_value else "N/A"
        prem = f"{bond.premium_rate:.1f}" if bond and bond.premium_rate else "N/A"
        
        info = f"""╔══════════════════════════════════════════════════════════════╗
║           可转债预测报告 - {code}                      ║
╚══════════════════════════════════════════════════════════════╝

【债券信息】
  名称: {bond_name}
  正股: {stock_code}
  规模: {issue_size}亿元
  评级: {rating}
  上市: {listing_date}

【转股指标】
  转股价值: {cv}
  溢价率:   {prem}%

【预测信息】
  市场情绪: {result.get('market_sentiment', 'N/A')}
  模型MAE:  ±{result.get('mae', 7.55):.2f}元
  预测区间: [{result.get('p20', 0):.1f}, {result.get('p80', 0):.1f}]元
"""
        self.result_text.insert(tk.END, info)
        
        # 预测结果（突出显示）
        pred_price = result['predicted_price']
        mae = result.get('mae', 7.55)
        pred = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                              ┃
┃        预测开盘价:   {pred_price:.1f} 元                              ┃
┃                                                              ┃
┃   预测区间: {pred_price - mae:.1f} ~ {pred_price + mae:.1f} 元 (置信度±{mae:.2f}元)        ┃
┃                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

子模型预测:
  ├─ 线性回归:     {result.get('lr', 0):.1f} 元
  ├─ K近邻:        {result.get('knn', 0):.1f} 元
  ├─ 梯度提升:     {result.get('gb', 0):.1f} 元
  └─ 分位数回归Q50: {result.get('q50', 0):.1f} 元
"""
        self.result_text.insert(tk.END, pred)
        
        # 实际对比
        if bond and bond.first_open:
            error = abs(pred_price - bond.first_open)
            pct = error / bond.first_open * 100
            
            if error < 3:
                judge = "✅ 非常准确"
            elif error < 8:
                judge = "✅ 误差可接受"
            else:
                judge = "⚠️ 误差较大"
            
            actual = f"""
【实际开盘对比】
  实际开盘: {bond.first_open:.1f} 元
  预测误差: {error:.1f} 元 ({pct:.1f}%)
  评价: {judge}
"""
            self.result_text.insert(tk.END, actual)
    
    def show_error(self, error_msg):
        self.predict_btn.config(state=tk.NORMAL)
        self.status_label.config(text="❌ 预测失败", foreground="red")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"❌ 预测出错\n\n{error_msg}")


def main():
    root = tk.Tk()
    CBPredictorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()