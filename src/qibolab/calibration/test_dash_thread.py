import qibolab.calibration.live as lp
import threading
import time
from qibolab.paths import qibolab_folder


#dash = multiprocessing.Process(target=lp.start_server())
path = qibolab_folder / 'calibration' / 'data' / 'buffer.npy'
dash = threading.Thread(target=lp.start_server, args=(path,))
dash.setDaemon(True)
dash.start()
for i in range(100):
    print("Do something")
    time.sleep(5)


