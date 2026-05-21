"""
マスタースクリプト：課題No.2のすべての処理を実行
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> bool:
    """
    スクリプトを実行
    """
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*60}")
    print(f"実行: {script_name}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"エラー: {script_name} の実行に失敗しました")
        print(f"リターンコード: {e.returncode}")
        return False
    except Exception as e:
        print(f"エラー: {e}")
        return False


def main() -> None:
    print("医用画像工学課題 No.2 - マスタースクリプト")
    print("="*60)
    
    scripts = [
        "01_differential_filters.py",
        "02_sharpening.py",
        "03_fft_filtering.py",
    ]
    
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    # 結果のサマリー
    print(f"\n{'='*60}")
    print("実行結果サマリー")
    print('='*60)
    
    for script, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{script}: {status}")
    
    all_success = all(results.values())
    print(f"\n全体状態: {'✓ すべて成功' if all_success else '✗ 一部失敗'}")
    
    if all_success:
        print("\n出力ファイルは outputs/ ディレクトリに保存されています。")


if __name__ == "__main__":
    main()
