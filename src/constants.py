# WANDB_ENTITY="zakhar-ostrovsky-team"
WANDB_ENTITY="geronua"

VALIDATION_ARTICLES =  [
    ('Nav1.4.pdf', 'Sodium channel protein type 4', 'human'),
    ('baranovskiy2014.pdf', 'DNA polymerase alpha', 'human'),
    ('kimura2019.pdf', '5-hydroxytryptamine receptor 2A', 'human'),
    ('kruse2015.pdf', '5-hydroxytryptamine receptor 2A', 'human'),
    ('kruse2015.pdf', 'Muscarinic acetylcholine receptor M2', 'human'),
    ('meng2021.pdf', 'Mixed lineage kinase domain-like protein', 'human'),
    ('murphy2014.pdf', 'Mixed lineage kinase domain-like protein', 'human'),
    ('nagar2006.pdf', 'Tyrosine-protein kinase ABL1', 'human'),
    ('simpson2019.pdf', 'Tyrosine-protein kinase ABL1', 'human'),
    ('liu2000.pdf', 'Dihydroorotate dehydrogenase', 'human'),
    ('evan2020.pdf', 'Dihydroorotate dehydrogenase', 'human'),
    ('maurer2012.pdf', 'GTPase KRas', 'human'),
    ('wang2021.pdf', 'GTPase KRas', 'human'),
    ('niu2019.pdf', 'Programmed cell death 1 ligand 1', 'human'),
    ('kim2020_.pdf', 'Gamma-aminobutyric acid receptor', 'human'),
]

TEST_ARTICLES = [
    ('Nav1.7.pdf', 'Sodium channel protein type 7', 'human'),
    ('baranovskiy2018.pdf', 'DNA polymerase alpha', 'human'),
    ('kim2020.pdf', '5-hydroxytryptamine receptor 2A', 'human'),
    ('kruse2012.pdf', 'Muscarinic acetylcholine receptor M2', 'human'),
    ('ma2017.pdf', 'Mixed lineage kinase domain-like protein', 'human'),
    ('das2013.pdf', 'Dihydroorotate dehydrogenase', 'human'),
    ('jingsong2011.pdf', 'Tyrosine-protein kinase ABL1', 'human'),
    ('weisberg2006.pdf', 'Tyrosine-protein kinase ABL1', 'human'),
    ('lim2021.pdf', 'GTPase KRas', 'human'),
    ('kang-pettinger2022.pdf', 'Programmed cell death 1 ligand 1', 'human'),
    ('chen2018.pdf', 'Gamma-aminobutyric acid receptor', 'human'),
]

ARTICLES_WITHOUT_POCKETS = [
    ('mcdermott2019.md', 'Sodium channel protein type 7'),
    ('nicole2021.md', 'Sodium channel protein type 4'),
    ('srivastava2003.md', 'DNA polymerase alpha'),
    ('perera2013.md', 'DNA polymerase alpha'),
    ('jayakumar2020.md', '5-hydroxytryptamine receptor 2A'),
    ('tanahashi2021.md', 'Muscarinic acetylcholine receptor M3'),
    ('ockenga2013.md', 'Muscarinic acetylcholine receptor M3'),
    ('lugo1990.md', 'Tyrosine-protein kinase ABL1'),
    ('zameitat2007.md', 'Dihydroorotate dehydrogenase'),
    ('barnes1993.md', 'Dihydroorotate dehydrogenase'),
    ('dusing2016.md', 'GTPase KRas'),
]

ARTICLES_WITH_POCKETS = [
    ('Nav1.7.md', 'Sodium channel protein type 7'),
    ('baranovskiy2014.md', 'DNA polymerase alpha'),
    ('kimura2019.md', '5-hydroxytryptamine receptor 2A'),
    ('kim2020.md', '5-hydroxytryptamine receptor 2A'),
    ('kruse2012.md', 'Muscarinic acetylcholine receptor M3'),
    ('kruse2015.md', 'Muscarinic acetylcholine receptor M2'),
    ('ma2017.md', 'Mixed lineage kinase domain-like protein'),
    ('murphy2014.md', 'Mixed lineage kinase domain-like protein'),
    ('jingsong2011.md', 'Tyrosine-protein kinase ABL1'),
    ('weisberg2006.md', 'Tyrosine-protein kinase ABL1'),
    ('simpson2019.md', 'Tyrosine-protein kinase ABL1'),
    ('liu2000.md', 'Dihydroorotate dehydrogenase'),
    ('das2013.md', 'Dihydroorotate dehydrogenase'),
    ('evan2020.md', 'Dihydroorotate dehydrogenase'),
    ('wang2021.md', 'GTPase KRas'),
    ('maurer2012.md', 'GTPase KRas'),
    ('chen2018.md', 'Gamma-aminobutyric acid receptor'),
]

ARTICLES_POCKETS = [
    ('shen2018.md', 'NaVPaS'),
    ('shen2017.md', 'NaVPaS'),
    ('Nav1.7.md', 'Sodium channel protein type 7'),
    ('mcdermott2019.md', 'Sodium channel protein type 7'), # No pockets
    ('baranovskiy2014.md', 'DNA polymerase alpha'),
    ('baranovskiy2018.md', 'DNA polymerase alpha'),
    ('srivastava2003.md', 'DNA polymerase alpha'), # No pockets
    ('perera2013.md', 'DNA polymerase alpha'), # No pockets
    ('kimura2019.md', '5-hydroxytryptamine receptor 2A'),
    ('kim2020.md', '5-hydroxytryptamine receptor 2A'),
    ('jayakumar2020.md', '5-hydroxytryptamine receptor 2A'), # No pockets
    ('kruse2012.md', 'Muscarinic acetylcholine receptor M3'),
    ('kruse2015.md', 'Muscarinic acetylcholine receptor M2'),
    ('tanahashi2021.md', 'Muscarinic acetylcholine receptor M3'), # No pockets
    ('ockenga2013.md', 'Muscarinic acetylcholine receptor M3'), # No pockets
    ('ma2017.md', 'Mixed lineage kinase domain-like protein'),
    ('murphy2014.md', 'Mixed lineage kinase domain-like protein'),
    ('jingsong2011.md', 'Tyrosine-protein kinase ABL1'),
    ('weisberg2006.md', 'Tyrosine-protein kinase ABL1'),
    ('simpson2019.md', 'Tyrosine-protein kinase ABL1'),
    ('lugo1990.md', 'Tyrosine-protein kinase ABL1'), # No pockets
    ('liu2000.md', 'Dihydroorotate dehydrogenase'),
    ('das2013.md', 'Dihydroorotate dehydrogenase'),
    ('evan2020.md', 'Dihydroorotate dehydrogenase'),
    ('zameitat2007.md', 'Dihydroorotate dehydrogenase'), # No pockets
    ('barnes1993.md', 'Dihydroorotate dehydrogenase'), # No pockets
    ('wang2021.md', 'GTPase KRas'),
    ('maurer2012.md', 'GTPase KRas'),
    ('dusing2016.md', 'GTPase KRas'), # No pockets
    ('chen2018.md', 'Gamma-aminobutyric acid receptor'),
    ('kim2020_.md', 'Gamma-aminobutyric acid receptor'),
]

ARTICLES_DEBUG = [
    ('shen2018.md', 'NaVPaS'),
    ('shen2017.md', 'NaVPaS'),
    ('murphy2014.md', 'Mixed lineage kinase domain-like protein'),
    ('zameitat2007.md', 'Dihydroorotate dehydrogenase'),
    ('barnes1993.md', 'Dihydroorotate dehydrogenase')
]
    # ('rook2011.md', 'Sodium channel protein type 5'),
    # ('baranovskiy2018.md', 'DNA polymerase alpha'),
    # ('kim2020_.md', 'Gamma-aminobutyric acid receptor'),
    # ('shen2018.md', 'Sodium channel protein type 5'),


    # ('kang-pettinger2022.md', 'Programmed cell death 1 ligand 1'),
    # ('ghosh2021.md', 'Programmed cell death 1 ligand 1'),
    # ('niu2019.md', 'Programmed cell death 1 ligand 1'),
    # ('doroshow2021.md', 'Programmed cell death 1 ligand 1'),