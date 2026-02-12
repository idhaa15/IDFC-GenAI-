import sys
import json
from utils.pipeline import process_invoice_single

def main():
    if len(sys.argv) < 2:
        print("Usage: python executable.py <image_path>")
        return

    img_path = sys.argv[1]
    result = process_invoice_single(img_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
