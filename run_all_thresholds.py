import subprocess

def main():
    for threshold in range(90, -1, -10):
        print(f"\n{'='*50}")
        print(f"RUNNING MAYMOTIF REPORT FOR {threshold}% THRESHOLD")
        print(f"{'='*50}\n")
        subprocess.run(["python", "MAYMOTIF-Reporting.py", str(threshold)])

if __name__ == "__main__":
    main()