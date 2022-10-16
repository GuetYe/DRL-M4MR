from pathlib import Path
from functools import reduce

WORK_DIR = Path.cwd().parent

# setting.py
FACTOR = 0.9  # the coefficient of 'bw' , 1 - FACTOR is the coefficient of 'delay'

METHOD = 'dijkstra'  # the calculation method of shortest path

DISCOVERY_PERIOD = 10  # discover network structure's period, the unit is seconds.

MONITOR_PERIOD = 5  # monitor period, bw

DELAY_PERIOD = 1.3  # detector period, delay

SCHEDULE_PERIOD = 6  # shortest forwarding network awareness period

PRINT_SHOW = False  # show or not show print

INIT_TIME = 30  # wait init for awareness

PRINT_NUM_OF_LINE = 8  # 一行打印8个值

LOGGER = True  # 是否保存日志

LINKS_INFO = WORK_DIR / "mininet/links_info/links_info.xml"  # 链路信息的xml文件路径

# SRC_IP = "10.0.0.1"
# DST_MULTICAST_IP = {'224.1.1.1': 1, }  # 组播地址： 标号（下面的索引号）
# DST_GROUP_IP = [["10.0.0.2", "10.0.0.3", "10.0.0.4"], ]  # 组成员的ip，（索引为上面的标号）

DST_MULTICAST_IP = {'224.1.1.1': ["10.0.0.2", "10.0.0.4", "10.0.0.11"], }  # 组播地址： 组成员的ip

WEIGHT = 'bw'
# FINAL_SWITCH_FLOW_IDEA = 1

finish_time_file = WORK_DIR / "mininet/finish_time.json"


def list_insert_one_by_one(list1, list2):
    l = []
    for x, y in zip(list1, list2):
        l.extend([x, y])
    return l


def gen_format_str(num):
    fmt = ''
    for i in range(num):
        fmt += '{{:<{}}}'
    # fmt += '\n'
    return fmt


# 只能打印key: value的两列，还不如用pandas
def print_pretty_table(param, titles, widths, table_name='zzlong', logger=None):
    """
        打印一个漂亮的表
    :param param: 要打印的字典，dict
    :param titles: 每列的title
    :param widths: 每列的宽度
    :param table_name: 表名字
    :param logger: 用什么打印 print / logger.info
    :return: None
    """
    f = logger if logger else print
    all_width = reduce(lambda x, y: x + y, widths)
    cut_line = "=" * all_width
    # 表名字
    w = all_width - len(table_name)
    if w > 1:
        f(cut_line[:w // 2] + table_name + cut_line[w // 2: w])
    else:
        f("=" + table_name + "=")

    # 以表格输出
    if isinstance(param, dict):
        # 获得{:^{}}多少个这个
        fmt = gen_format_str(len(titles))
        # 确定宽度
        width_fmt = fmt.format(*widths)
        # 确定值
        title_fmt = width_fmt.format(*titles)
        # 打印第一行title
        f(title_fmt)
        # 打印分割线
        f(cut_line)
        # 打印每一行的值
        for k, v in param.items():
            content_fmt = width_fmt.format(str(k), str(v))
            # 打印内容
            f(content_fmt)

    # 打印分割线
    f(cut_line + '\n')


# def print_pretty_list(param, num, width=10, table_name='zzlong', logger=None):
#     """
#         按每行固定个，打印列表中的值
#     :param param: 要打印的列表 list
#     :param num: 每行多少个值
#     :param width: 每个值的宽度
#     :param table_name: 表名字
#     :param logger: 用什么打印 print / logger.info
#     :return: None
#     """
#     f = logger if logger else print
#     all_widths = num * width
#     cut_line = "=" * all_widths
#     # 表名字
#     w = all_widths - len(table_name)
#     if w > 1:
#         f(cut_line[:w // 2] + table_name + cut_line[w // 2: w])
#     else:
#         f("=" + table_name + "=")
#
#     # 直接打印
#     temp = 0
#     for i in range(len(param) // num):
#         f(param[temp: temp + num])
#         temp += num
#     if param[temp:]:
#         f(param[temp:])
#     else:
#         pass
#
#     # 打印分割线
#     f(cut_line + '\n')


if __name__ == '__main__':
    # a = {'test1': [11, 12, 13], 'test2': [21, 22, 23], 'test3': [31, 32, 33]}
    # print_pretty_table(a, ['my_test', 'values'], [10, 14], 'test_table', print)
    #
    # b = list(range(30))
    # print_pretty_list(b, 10, 10)
    print(WORK_DIR)
    print(LINKS_INFO)
    print(finish_time_file)
