"""
检查 EquiRating 项目依赖是否正确安装
"""

import sys

def check_package(package_name, import_name=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        print(f'✓ {package_name:<20} {version}')
        return True
    except ImportError:
        print(f'✗ {package_name:<20} 未安装')
        return False

print('=' * 60)
print('EquiRating 依赖检查')
print('=' * 60)
print()

# 检查所有依赖
packages = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('scikit-learn', 'sklearn'),
    ('xgboost', 'xgboost'),
    ('joblib', 'joblib'),
    ('beautifulsoup4', 'bs4'),
    ('lxml', 'lxml'),
    ('openpyxl', 'openpyxl'),
]

results = []
for pkg_name, import_name in packages:
    results.append(check_package(pkg_name, import_name))

print()
print('=' * 60)
if all(results):
    print('✓ 所有依赖已正确安装！')
else:
    print('✗ 部分依赖未安装，请运行：')
    print('  pip install -r requirements.txt')
print('=' * 60)
