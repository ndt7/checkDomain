"""Generates data for train/test algorithms"""
from datetime import datetime
from urllib.request import urlopen

import pickle
import os
import random

''''tldextract:  dung de trich xuat ten mien tu url'''
import tldextract
import pandas as pd

from lib import lockyv2, cryptolocker, corebot, pykspa, ramnit, banjori, ramdo, qakbot, dircrypt, simda, kraken

# Location of Alexa 1M
ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

# Our ourput file containg all the training data
'''Tệp đầu ra của chúng tôi chứa tất cả dữ liệu đào tạo'''
DATA_FILE = 'traindata.pkl'

"""kho nhat la xu ly du lieu """
def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):
    """Grabs Alexa 1M"""
    url = urlopen(address)
    #zipfile = ZipFile(StringIO(url.read()))
    zipfile = pd.read_csv("top-1m.csv")
    return [tldextract.extract(x.split(',')[1]).domain for x in \
            zipfile.read(filename).split()[:num]]



def gen_malicious(num_per_dga=10000):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []

    '''
     We use some arbitrary seeds to create domains with banjori
     Chúng tôi sử dụng một số hạt giống tùy ý để tạo tên miền với banjori
    '''
    banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                     'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                     'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                     'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                     'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                     'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                     'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                     'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                     'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                     'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                     'atlanta', 'portland', 'seattle', 'washingtondc']

    segs_size = max(1, num_per_dga / len(banjori_seeds))
    for banjori_seed in banjori_seeds:
        domains += banjori.generate_domains(segs_size, banjori_seed) # ??

        # thu vien //
        # fix bug https://www.freecodecamp.org/news/typeerror-cant-multiply-sequence-by-non-int-of-type-float-solved/
        labels += ['banjori'] * int(segs_size)

    domains += corebot.generate_domains(num_per_dga)
    labels += ['corebot'] * num_per_dga

    # Create different length domains using cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = max(1, num_per_dga / len(crypto_lengths))
    for crypto_length in crypto_lengths:
        domains += cryptolocker.generate_domains(segs_size,
                                                 seed_num=random.randint(
                                                     1, 1000000),
                                                 length=crypto_length)
        labels += ['cryptolocker'] * int(segs_size)

    domains += dircrypt.generate_domains(num_per_dga)
    labels += ['dircrypt'] * num_per_dga

    # generate kraken and divide between configs
    kraken_to_gen = max(1, num_per_dga / 2)
    domains += kraken.generate_domains(kraken_to_gen,
                                       datetime(2016, 1, 1), 'a', 3)
    labels += ['kraken'] * int(kraken_to_gen)
    domains += kraken.generate_domains(kraken_to_gen,
                                       datetime(2016, 1, 1), 'b', 3)
    labels += ['kraken'] * int(kraken_to_gen)

    # generate locky and divide between configs
    locky_gen = max(1, num_per_dga / 11)
    for i in range(1, 12):
        domains += lockyv2.generate_domains(locky_gen, config=i)
        labels += ['locky'] * int(locky_gen)

    # Generate pyskpa domains
    domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    labels += ['pykspa'] * num_per_dga

    # Generate qakbot
    domains += qakbot.generate_domains(num_per_dga, tlds=[])
    labels += ['qakbot'] * num_per_dga

    # ramdo divided over different lengths
    ramdo_lengths = range(8, 32)
    segs_size = max(1, num_per_dga / len(ramdo_lengths))
    for rammdo_length in ramdo_lengths:
        domains += ramdo.generate_domains(segs_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=rammdo_length)
        labels += ['ramdo'] * int(segs_size)

    # ramnit
    domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
    labels += ['ramnit'] * num_per_dga

    # simda
    simda_lengths = range(8, 32)
    segs_size = max(1, num_per_dga / len(simda_lengths))
    for simda_length in range(len(simda_lengths)):
        domains += simda.generate_domains(segs_size,
                                          length=simda_length,
                                          tld=None,
                                          base=random.randint(2, 2 ** 32))
        labels += ['simda'] * int(segs_size)

    return domains, labels


def gen_data(force=False):
    """Grab all data for train/test and save

    force:If true overwrite, else skip if file
          already exists
    """
    if force or (not os.path.isfile(DATA_FILE)):
        domains, labels = gen_malicious(10000)

        # Get equal number of benign/malicious
        domains += get_alexa(len(domains))
        labels += ['benign'] * len(domains)

        pickle.dump(zip(labels, domains), open(DATA_FILE, 'w'))


# load data // lay du lieu

''' tra ve du lieu va label '''
"""data.get_data"""
def get_data(force=False):
    """Returns data and labels"""
    ''' lay du lieu tra ve '''
    gen_data(force)
    ''' tra ve '''
    return pickle.load(open(DATA_FILE))
