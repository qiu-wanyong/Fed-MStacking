import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pywt
import random
import matplotlib.cm as cm

# def plot_signal(audio_data, title=None):
#     plt.figure()
#     plt.plot(audio_data, linewidth=1)
#     plt.title(title,fontsize = 16)
#     plt.tick_params(labelsize=12)
#     plt.grid()
#     plt.show()

def CWT(data, fs=25600):
    t = np.arange(0, len(data)) / fs
    wavename = "gaus4"   # cgau8 小波
    # wavename = "morl"  # morlet 小波
    # wavename = "cmor3-3"  # cmor 小波

    totalscale = 256
    fc = pywt.central_frequency(wavename)  # 中心频率
    cparam = 2 * fc * totalscale
    scales = cparam / np.arange(totalscale, 1, -1)
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换
    plt.figure(figsize=(18, 12))
    # ax1 = plt.subplot(1,2,1)
    # plt.plot(t, data)
    # plt.xlabel("Time(s)", fontsize = 14)
    # plt.ylabel("Amplitude(g)", fontsize=14)
    # ax2 = plt.subplot(1,2,2)
    plt.contourf(t, frequencies, abs(cwtmatr))  # 画等高线图

    # yt = [15.625, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
    plt.yscale('log')
    # plt.yticks(yt)

    # print("min(frequencies):", min(frequencies))
    # print("max(frequencies):", max(frequencies))
    plt.ylim([min(frequencies), max(frequencies)])

    plt.xlabel("Time(s)", fontsize = 40)
    plt.ylabel("Frequency(Hz)", fontsize=40)
    plt.tick_params(labelsize=40)
    plt.tight_layout()
    plt.savefig("abnormal_cwt.jpg", dpi=600, format="jpg")
    # plt.savefig("normal_cwt.svg", dpi=600, format="svg")
    plt.show()

path = 'E:/GithubSpace/JBHI_heart_data/source audio/training-a/a0010.wav'
y2, sr = librosa.load(path, sr = None)
CWT(y2, sr)

# print(sr)
# 感觉这个画音频图更好
# plt.figure(figsize=(14, 5))
# # 调用librosa包画出波形图
# librosa.display.waveshow(y2, sr=sr)
# # 设置画布标题
# plt.xlabel("Time/s", fontsize = 14)
# plt.ylabel("Amplitude/dB", fontsize=14)
# #plt.title('sound wave')
# # 显示画布
# plt.show()
plt.figure(figsize=(18, 12))
# 调用librosa包画出波形图
librosa.display.waveshow(y2, sr=sr)
# 设置画布标题\
plt.xlabel('Time/s', fontsize=40)
plt.ylabel('Amplitude/dB', fontsize=40)
plt.tick_params(labelsize=40)
plt.grid()
# plt.savefig("normal.svg", dpi=300,format="svg")
plt.show()

#音频法2
# plot_signal(y, title='Initial Audio')
start = random.randint(0, int(len(y2)/sr)-5)
duration = 5
stop = start +duration
y = y2[start*sr:stop*sr]
wavename = 'gaus4'
totalscal = 256  # totalscal是对信号进行小波变换时所用尺度序列的长度(通常需要预先设定好)
fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
cparam = 2 * fc * totalscal  # 常数c
scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
[cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / sr)
#t = np.arange(0, y.shape[0]/sr, 1.0/sr)
t = np.arange(0, len(y)) / sr
plt.contourf(t, frequencies, abs(cwtmatr))
# plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
#yt = [15.625, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
plt.yscale('log')
plt.xticks([])
plt.yticks([])
plt.yticks([])
plt.axis('off')
# plt.yticks(yt)
# plt.ylim([min(frequencies), max(frequencies)])
plt.show()


# for wavename in pywt.wavelist(kind='continuous'):
#     for totalscal in [4, 32, 128, 256]:
#         fc = pywt.central_frequency(wavename)  # 计算小波函数的中心频率
#         cparam = 2 * fc * totalscal  # 常数c
#         scales = cparam / np.arange(totalscal, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）
#         [cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / sr)
#         # t = np.arange(0, y.shape[0]/sr, 1.0/sr)
#         t = np.arange(0, len(y)) / sr
#         plt.contourf(t, frequencies, abs(cwtmatr))
#         plt.ylabel(u"freq(Hz)")
#         plt.xlabel(u"time(s)")
#         plt.subplots_adjust(hspace=0.4)  # 调整边距和子图的间距 hspace为子图之间的空间保留的高度，平均轴高度的一部分
#         plt.title = ("小波时频图")
#         # yt = [15.625, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
#         plt.yscale('log')
#         # plt.yticks(yt)
#         plt.ylim([min(frequencies), max(frequencies)])
#         plt.savefig("image for cwt/" + wavename + str(totalscal) + ".png")
#         plt.clf()
#         plt.close()

# plt.figure(figsize=(12, 6))
# plt.contourf(t, frequencies, abs(cwtmatr))
# 一些铺平的操作
# plt.axis('off')
# plt.gcf().set_size_inches(784 / 100, 784 / 100)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.show()

# wavename = 'cgau8'
# totalscal = 256
# fc = pywt.central_frequency(wavename)
# cparam = 2 * fc * totalscal
# scales = cparam / np.arange(totalscal, 1, -1)
# [cwtmatr, frequencies] = pywt.cwt(y, scales, wavename, 1.0 / sr)
# plt.figure(figsize=(8, 4))
# plt.contourf(t, frequencies, abs(cwtmatr))
# plt.ylabel(u"频率(Hz)")
# plt.xlabel(u"时间(秒)")
# plt.subplots_adjust(hspace=0.4)
# plt.ylim([min(frequencies), max(frequencies)])
# plt.show()

# 三d连续小波变换结果
# X = t
# Y = frequencies
# Z = abs(cwtmatr)
# X, Y = np.meshgrid(X, Y)
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)
# # ax.set_zlim(60, 170)
# ax.set_xlabel('time')
# ax.set_ylabel('freq')
# ax.set_zlabel('db')
# fig.colorbar(surf, shrink=0.5, aspect=20)
# plt.show()


"""
连续小波变换 CWT
参考论文：https://www.mdpi.com/2076-3417/8/7/1102/html
morlet 小波在轴承故障诊断中比较常用
"""
#
# def CWT(data, fs=25600):
#     t = np.arange(0, len(data)) / fs
#     # wavename = "cgau8"   # cgau8 小波
#     wavename = "morl"  # morlet 小波
#     # wavename = "cmor3-3"  # cmor 小波
#
#     totalscale = 256
#     fc = pywt.central_frequency(wavename)  # 中心频率
#     cparam = 2 * fc * totalscale
#     scales = cparam / np.arange(totalscale, 1, -1)
#     [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / fs)  # 连续小波变换
#     plt.figure(figsize=(12, 6))
#     ax1 = plt.subplot(1,2,1)
#     plt.plot(t, data)
#     plt.xlabel("Time(s)", fontsize = 14)
#     plt.ylabel("Amplitude(g)", fontsize=14)
#     ax2 = plt.subplot(1,2,2)
#     plt.contourf(t, frequencies, abs(cwtmatr))  # 画等高线图
#
#     yt = [15.625, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
#     ax2.set_yscale('log')
#     ax2.set_yticks(yt)
#     ax2.set_yticklabels(yt)
#
#     # print("min(frequencies):", min(frequencies))
#     # print("max(frequencies):", max(frequencies))
#     ax2.set_ylim([min(frequencies), max(frequencies)])
#
#     plt.xlabel("Time(s)", fontsize = 14)
#     plt.ylabel("Frequency(Hz)", fontsize=14)
#     plt.title(file_name, fontsize=14 )
#     plt.tight_layout()
#     plt.savefig("./cwt_figures/" + file_name + "_CWT" + ".png")
#     # plt.show()
#
#
#
# def gener_simul_data():
#     fs = 1024
#     t = np.arange(0, 1.0, 1.0 / fs)
#     f1 = 100
#     f2 = 200
#     f3 = 300
#     data = np.piecewise(t, [t<1, t<0.8, t<0.3],
#                         [lambda t: np.sin(2 * np.pi * f1 * t),
#                          lambda t: np.sin(2 * np.pi * f2 * t),
#                          lambda t: np.sin(2 * np.pi * f3 * t)])
#     return data
#
# if __name__ == "__main__":
#     print(pywt.families())
#     print(pywt.wavelist('morl'))
#     file_path = "../raw_data/Training_set/Bearing1_1/acc/"
#
#     file_list = os.listdir(file_path)
#     print("file_path:", file_path)
#     print("num files:", len(file_list))
#
#     for file_name in file_list[:]:
#         file_dir = file_path + file_name
#         csv_data = pd.read_csv(file_dir, header=None)  # 无表头的表格
#         data_h = np.array(csv_data.iloc[0:, -2].tolist())
#         data_v = np.array(csv_data.iloc[0:, -1].tolist())
#
#         # data = gener_simul_data()
#
#         CWT(data_h, fs=25600)