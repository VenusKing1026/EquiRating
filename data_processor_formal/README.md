# data_processor_formal（正式版）

## 目录约定
- `code/`：所有脚本代码
- `data/<year>/raw_html/`：该年份原始 HTML（包含 `TOPmapping.html` 和 `TOP20.html`）
- `data/<year>/players_mapping.csv`：旧逻辑产出的名字-网站映射表
- `data/<year>/players.csv`：新逻辑产出的当年 Top 顺序+名字+网站
- `data/<year>/processed_data/`：处理结果

## 脚本命名
- `get_top_mapping.py`：旧 get_top 逻辑（从 `TOPmapping.html` 取名字+网站）
- `get_top_rank.py`：新 get_top 逻辑（从 `TOP20.html` 取顺序+名字，再查 mapping 回填网站）
- `top_workflow.py`：整合流程（先 mapping，再 rank）

## 最简运行流程
在项目根目录执行：

1. 一键生成 `players_mapping.csv` + `players.csv`
```bash
python data_processor_formal/code/top_workflow.py --year 2026
```

2. 批量打开选手页面并处理该年份 `raw_html` 下全部 HTML
```bash
python data_processor_formal/code/batch_workflow.py --year 2026
```

## 批处理自动整合（可选）
如果希望 `batch_workflow` 内自动先跑 `top_workflow`：
```bash
python data_processor_formal/code/batch_workflow.py --year 2026 --prepare-players
```

## 已保存 body 时直接处理（不打开浏览器）
```bash
python data_processor_formal/code/batch_workflow.py --year 2026 --process-only
```

## 单文件处理（可选）
```bash
python data_processor_formal/code/data_processor.py --year 2026 --input-html "data_processor_formal/data/2026/raw_html/你的文件.html"
```
