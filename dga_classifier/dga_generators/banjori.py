# Original code: https://github.com/baderj/domain_generation_algorithms/blob/master/banjori/dga.py

def map_to_lowercase_letter(s):
    # ord tra ve 1 so nguyen cho 1 ky tu
    return ord('a') + ((s - ord('a')) % 26)


def next_domain(domain):
    # thuc hien
    # 1 danh sach ord(x)
    # duyet lan luot (domain) roi luu vao dl
    dl = [ord(x) for x in list(domain)]
    dl[0] = map_to_lowercase_letter(dl[0] + dl[3])
    dl[1] = map_to_lowercase_letter(dl[0] + 2 * dl[1])
    dl[2] = map_to_lowercase_letter(dl[0] + dl[2] - 1)
    dl[3] = map_to_lowercase_letter(dl[1] + dl[2] + dl[3])
    return ''.join([chr(x) for x in dl])  # cac ky tu duoc noi lai 
    # return ve chu cai tu so nguyen ma chu cai dai dien 


def generate_domains(nr_domains, seed='hereisaseeddomainthatweuse.com'):
    ret = []
    for i in range(int(nr_domains)):
        seed = next_domain(seed)
        ret.append(seed)
    return ret
