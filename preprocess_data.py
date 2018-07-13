import os
import re
import argparse


def convert_to_bis(source_dir, target_path, append=True):
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            file = os.path.join(root, name)
            bises = process_file(file)
            if append:
                _save_bises(bises, target_path, write_mode='a')
            else:
                _save_bises(bises, target_path)


def _save_bises(bises, path, write_mode='w'):
    with open(path, write_mode, encoding='UTF-8') as f:
        for bis in bises:
            for char, tag in bis:
                f.write(char + ' ' + tag + '\n')
            f.write('\n')


def process_file(file):
    with open(file, 'r', encoding='UTF-8') as f:
        text = f.readlines()
        bises = _parse_text(text)
    return bises


def _parse_text(text: list):
    bises = []
    for line in text:
        # remove POS tag
        line, _ = re.subn('/[a-z]+[0-9]?|\\n', '', line)
        if line == '\n':
            continue
        bises.append(_tag(line))
    return bises


def _tag(line):
    """
    给指定的一行文本打上BIS标签
    :param line: 文本行
    :return:
    """
    bis = []
    words = re.split('\s+', line)
    words = list(map(list, words))
    pre_word = None
    for word in words:
        if len(word) == 0:
            continue
        if word[0] == '[':
            pre_word = word
            continue
        if pre_word is not None:
            pre_word += word
            if word[-1] != ']':
                continue
            else:
                word = pre_word[1: - 1]
                pre_word = None

        if len(word) == 1:
            bis.append((word[0], 'S'))
        else:
            for i, char in enumerate(word):
                if i == 0:
                    bis.append((char, 'B'))
                else:
                    bis.append((char, 'I'))
    # bis.append(('\n', 'O'))
    return bis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将使用词性标注的文件转换为用BIS分块标记的文件。")
    parser.add_argument("corups_dir", type=str, help="指定存放语料库的文件夹，程序将会递归查找目录下的文件。")
    parser.add_argument("output_path", type=str, default='.', help="指定标记好的文件的输出路径。")
    parser.add_argument("-a", "--append", help="写入文件的模式为追加", action="store_true")
    args = parser.parse_args()

    print("Converting...")
    convert_to_bis(args.corups_dir, args.output_path, args.append)
    print("Converted.")
