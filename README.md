# 期末專案：Hermes-DQN

## 📌 專案總覽
**Hermes-DQN: Memory-Augmented LLM Framework for Automated Reinforcement Learning Reward Design**
*(Hermes-DQN：用於自動化強化學習獎勵函數設計的記憶擴增大型語言模型框架)*

本專案探討如何利用具備記憶體擴增能力的大型語言模型 (LLMs)，為深度強化學習 (DRL) 演算法（特別針對 DQN 相關架構）提供自動化的獎勵機制設計。

## 📺 系統介紹影片
點擊下方圖片即可觀看完整的 YouTube 系統介紹影片：

[![系統介紹影片](https://img.youtube.com/vi/FmPRpDJ9JXI/maxresdefault.jpg)](https://youtu.be/FmPRpDJ9JXI)

> 🎥 **提示**：如果預覽圖沒有顯示，您也可以透過[這條連結直接觀看影片](https://youtu.be/FmPRpDJ9JXI)。

## 🏗️ 系統架構圖

![Hermes-DQN 系統架構圖](images/architecture.png)

*Hermes-DQN 整體系統架構，包含教授系統（Nous Research Hermes Agent）、教授的大腦（Google Gemma 4 31B API）及球員管理（DQN Agent & Gymnasium Env）三大核心模組。*

## 📁 專案架構與檔案目錄
- `/PPT`
  - `Hermes-DQN Memory-Augmented LLM Framework for Automated Reinforcement Learning Reward Design.pdf` （完整報告與方法論）
  - `Hermes-DQN_Introduction_正式版.pdf` （簡介簡報）
- `/images`
  - `architecture.png` （系統架構圖）
- `/docx`
  - `Hermes-DQN_Paper_Chapters_1-3_v2.docx` （論文草稿）
- 其他相關原始碼、環境設定與測試腳本將結構化整理於此儲存庫中。
