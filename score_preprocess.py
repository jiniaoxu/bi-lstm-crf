from preprocess_data import *


def remove_pos(source_dir, target_path):
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            file = os.path.join(root, name)
            bises = process_file(file)
            lines = _convert_to_lines(bises)
            save_to(lines, os.path.join(target_path))


def restore(source_dir, target_path):
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            file = os.path.join(root, name)
            bises = process_file(file)
            lines = _convert_to_lines(bises)
            save_to(lines, os.path.join(target_path), with_white=False)


def _convert_to_lines(bises):
    lines = []
    for bia in bises:
        line, t1 = [], []
        for c, t in bia:
            if t == 'B':
                if len(t1) != 0:
                    line.append(t1)
                t1 = [c]
            if t == 'I':
                t1.append(c)
            if t == 'S':
                if len(t1) != 0:
                    line.append(t1)
                t1 = []
                line.append([c])
        lines.append(line)
    return lines


def save_to(lines, file_path, with_white=True):
    with open(file_path, 'a', encoding='UTF-8') as f:
        for line in lines:
            line_str = list(map(lambda words: ''.join(words), line))
            if with_white:
                f.write(' '.join(line_str) + '\n')
            else:
                f.write(''.join(line_str) + '\n')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="根据指定的语料生成黄金标准文件与其相应的无分词标记的原始文件")
    parse.add_argument("--corups_dir", help="语料文件夹", default="./corups/2014")
    parse.add_argument("--gold_file_path", help="生成的黄金标准文件路径", default="./corups/filter/gold.utf8")
    parse.add_argument("--restore_file_path", help="生成无标记的原始文件路径", default="./corups/filter/restore.utf8")

    args = parse.parse_args()
    corups_dir = args.corups_dir
    gold_file_path = args.corups_file_path
    restore_file_path = args.restore_file_path

    print("Processing...")
    remove_pos(corups_dir, gold_file_path)
    restore(corups_dir, restore_file_path)
    print("Process done.")
