import os
import scipy.io
import psutil

# === 1. Paths ===
dataDir = '/home/rhyscornelious/neural_seq_decoder/data/competitionData'

# Make sure sessionNames is defined
try:
    print("Session names:")
    print(sessionNames)
except NameError:
    print("sessionNames variable not defined. Please define it before running this script.")

# === 2. Focus on dayIdx = 5 ===
dayIdx = 5
if 'sessionNames' in globals() and len(sessionNames) > dayIdx:
    train_path = os.path.join(dataDir, 'train', sessionNames[dayIdx] + '.mat')
    test_path = os.path.join(dataDir, 'test', sessionNames[dayIdx] + '.mat')

    print("\nPaths for dayIdx=5:")
    print("Train:", train_path)
    print("Test:", test_path)

    # File sizes
    if os.path.exists(train_path):
        print("Train file size:", os.path.getsize(train_path)/1e6, "MB")
    else:
        print("Train file not found.")

    if os.path.exists(test_path):
        print("Test file size:", os.path.getsize(test_path)/1e6, "MB")
    else:
        print("Test file not found.")

    # === 3. .mat file keys and shapes ===
    try:
        train_data = scipy.io.loadmat(train_path)
        test_data = scipy.io.loadmat(test_path)
        print("\nTrain .mat keys:", train_data.keys())
        for k in train_data.keys():
            try:
                print(k, getattr(train_data[k], 'shape', None))
            except:
                pass
        print("\nTest .mat keys:", test_data.keys())
        for k in test_data.keys():
            try:
                print(k, getattr(test_data[k], 'shape', None))
            except:
                pass
    except Exception as e:
        print("Error loading .mat files:", e)

# === 4. System memory ===
mem = psutil.virtual_memory()
print("\nSystem memory:")
print(mem)

# === 5. CPU usage snapshot ===
print("\nCPU usage:")
print(psutil.cpu_percent(interval=1), "%")

# === 6. Disk usage ===
print("\nDisk usage:")
for path in [dataDir, '/home/rhyscornelious', '/']:
    usage = shutil.disk_usage(path)
    print(path, "Total:", usage.total/1e9, "GB, Used:", usage.used/1e9, "GB, Free:", usage.free/1e9, "GB")
