import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

from Bio import Phylo
from ete3 import Tree, TreeNode

DATE = 'date'
DATE_CI = 'date_CI'

DATE_REGEX = r'[+-]*[\d]+[.\d]*(?:[e][+-][\d]+){0,1}'
DATE_COMMENT_REGEX = '[&,:]date[=]["]{{0,1}}({})["]{{0,1}}'.format(DATE_REGEX)
CI_DATE_REGEX_LSD = '[&,:]CI_date[=]["]{{0,1}}({}) ({})["]{{0,1}}'.format(DATE_REGEX, DATE_REGEX)
CI_DATE_REGEX_PASTML = '[&,:]date_CI[=]["]{{0,1}}({})[|]({})["]{{0,1}}'.format(DATE_REGEX, DATE_REGEX)
COLUMN_REGEX_PASTML = '[&,]{column}[=]([^]^,]+)'

IS_POLYTOMY = 'polytomy'


def get_dist_to_root(tip):
    dist_to_root = 0
    n = tip
    while not n.is_root():
        dist_to_root += n.dist
        n = n.up
    return dist_to_root


def annotate_dates(forest, root_dates=None):
    if root_dates is None:
        root_dates = [0] * len(forest)
    for tree, root_date in zip(forest, root_dates):
        for node in tree.traverse('preorder'):
            if getattr(node, DATE, None) is None:
                if node.is_root():
                    node.add_feature(DATE, root_date if root_date else 0)
                else:
                    node.add_feature(DATE, getattr(node.up, DATE) + node.dist)
            else:
                node.add_feature(DATE, float(getattr(node, DATE)))
            ci = getattr(node, DATE_CI, None)
            if ci and not isinstance(ci, list) and not isinstance(ci, tuple):
                node.del_feature(DATE_CI)
                if isinstance(ci, str) and '|' in ci:
                    try:
                        node.add_feature(DATE_CI, [float(_) for _ in ci.split('|')])
                    except:
                        pass


def name_tree(tree, suffix=""):
    """
    Names all the tree nodes that are not named or have non-unique names, with unique names.

    :param tree: tree to be named
    :type tree: ete3.Tree

    :return: void, modifies the original tree
    """
    existing_names = Counter()
    n_nodes = 0
    for _ in tree.traverse():
        n_nodes += 1
        if _.name:
            existing_names[_.name] += 1
            if '.polytomy_' in _.name:
                _.add_feature(IS_POLYTOMY, 1)
    if n_nodes == len(existing_names):
        return
    i = 0
    new_existing_names = Counter()
    for node in tree.traverse('preorder'):
        name_prefix = node.name if node.name and existing_names[node.name] < 10 \
            else '{}{}{}'.format('t' if node.is_leaf() else 'n', i, suffix)
        name = 'root{}'.format(suffix) if node.is_root() else name_prefix
        while name is None or name in new_existing_names:
            name = '{}{}{}'.format(name_prefix, i, suffix)
            i += 1
        node.name = name
        new_existing_names[name] += 1


def collapse_zero_branches(forest, features_to_be_merged=None):
    """
    Collapses zero branches in tre tree/forest.

    :param forest: tree or list of trees
    :type forest: ete3.Tree or list(ete3.Tree)
    :param features_to_be_merged: list of features whose values are to be merged
        in case the nodes are merged during collapsing
    :type features_to_be_merged: list(str)
    :return: void
    """
    num_collapsed = 0

    if features_to_be_merged is None:
        features_to_be_merged = []

    for tree in forest:
        for n in list(tree.traverse('postorder')):
            zero_children = [child for child in n.children if not child.is_leaf() and child.dist <= 0]
            if not zero_children:
                continue
            for feature in features_to_be_merged:
                feature_intersection = set.intersection(*(getattr(child, feature, set()) for child in zero_children)) \
                                       & getattr(n, feature, set())
                if feature_intersection:
                    value = feature_intersection
                else:
                    value = set.union(*(getattr(child, feature, set()) for child in zero_children)) \
                            | getattr(n, feature, set())
                if value:
                    n.add_feature(feature, value)
            for child in zero_children:
                n.remove_child(child)
                for grandchild in child.children:
                    n.add_child(grandchild)
            num_collapsed += len(zero_children)
    if num_collapsed:
        logging.getLogger('pastml').debug('Collapsed {} internal zero branches.'.format(num_collapsed))


def remove_certain_leaves(tr, to_remove=lambda node: False):
    """
    Removes all the branches leading to leaves identified positively by to_remove function.
    :param tr: the tree of interest (ete3 Tree)
    :param to_remove: a method to check is a leaf should be removed.
    :return: void, modifies the initial tree.
    """

    tips = [tip for tip in tr if to_remove(tip)]
    for node in tips:
        if node.is_root():
            return None
        parent = node.up
        parent.remove_child(node)
        # If the parent node has only one child now, merge them.
        if len(parent.children) == 1:
            brother = parent.children[0]
            brother.dist += parent.dist
            if parent.is_root():
                brother.up = None
                tr = brother
            else:
                grandparent = parent.up
                grandparent.remove_child(parent)
                grandparent.add_child(brother)
    return tr


def read_forest(tree_path, columns=None):
    try:
        roots = parse_nexus(tree_path, columns=columns)
        if roots:
            return roots
    except:
        pass
    with open(tree_path, 'r') as f:
        nwks = f.read().replace('\n', '').split(';')
    if not nwks:
        raise ValueError('Could not find any trees (in newick or nexus format) in the file {}.'.format(tree_path))
    return [read_tree(nwk + ';', columns) for nwk in nwks[:-1]]


def read_tree(tree_path, columns=None):
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(tree_path, format=f)
            break
        except:
            continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))
    if columns:
        for n in tree.traverse():
            for c in columns:
                vs = set(getattr(n, c).split('|')) if hasattr(n, c) else set()
                if vs:
                    n.add_feature(c, vs)
    return tree


def parse_nexus(tree_path, columns=None):
    trees = []
    for nex_tree in read_nexus(tree_path):
        todo = [(nex_tree.root, None)]
        tree = None
        while todo:
            clade, parent = todo.pop()
            dist = 0
            try:
                dist = float(clade.branch_length)
            except:
                pass
            node = TreeNode(dist=dist, name=getattr(clade, 'name', None))
            if parent is None:
                tree = node
            else:
                parent.add_child(node)

            # Parse LSD2 dates and CIs, and PastML columns
            date, ci = None, None
            columns2values = defaultdict(set)
            comment = getattr(clade, 'comment', None)
            if isinstance(comment, str):
                date = next(iter(re.findall(DATE_COMMENT_REGEX, comment)), None)
                ci = next(iter(re.findall(CI_DATE_REGEX_LSD, comment)), None)
                if ci is None:
                    ci = next(iter(re.findall(CI_DATE_REGEX_PASTML, comment)), None)
                if columns:
                    for column in columns:
                        values = \
                            set.union(*(set(_.split('|')) for _ in re.findall(COLUMN_REGEX_PASTML.format(column=column),
                                                                              comment)), set())
                        if values:
                            columns2values[column] |= values
            comment = getattr(clade, 'branch_length', None)
            if not ci and not parent and isinstance(comment, str):
                ci = next(iter(re.findall(CI_DATE_REGEX_LSD, comment)), None)
                if ci is None:
                    ci = next(iter(re.findall(CI_DATE_REGEX_PASTML, comment)), None)
            comment = getattr(clade, 'confidence', None)
            if ci is None and comment is not None and isinstance(comment, str):
                ci = next(iter(re.findall(CI_DATE_REGEX_LSD, comment)), None)
                if ci is None:
                    ci = next(iter(re.findall(CI_DATE_REGEX_PASTML, comment)), None)
            if date is not None:
                try:
                    date = float(date)
                    node.add_feature(DATE, date)
                except:
                    pass
            if ci is not None:
                try:
                    ci = [float(_) for _ in ci]
                    node.add_feature(DATE_CI, ci)
                except:
                    pass
            if columns2values:
                for c, vs in columns2values.items():
                    node.add_feature(c, vs)
            todo.extend((c, node) for c in clade.clades)
        for n in tree.traverse('preorder'):
            date, ci = getattr(n, DATE, None), getattr(n, DATE_CI, None)
            if date is not None or ci is not None:
                for c in n.children:
                    if c.dist == 0:
                        if getattr(c, DATE, None) is None:
                            c.add_feature(DATE, date)
                        if getattr(c, DATE_CI, None) is None:
                            c.add_feature(DATE_CI, ci)
        for n in tree.traverse('postorder'):
            date, ci = getattr(n, DATE, None), getattr(n, DATE_CI, None)
            if not n.is_root() and n.dist == 0 and (date is not None or ci is not None):
                if getattr(n.up, DATE, None) is None:
                    n.up.add_feature(DATE, date)
                if getattr(n.up, DATE_CI, None) is None:
                    n.up.add_feature(DATE_CI, ci)

        # propagate dates up to the root if needed
        if getattr(tree, DATE, None) is None:
            dated_node = next((n for n in tree.traverse() if getattr(n, DATE, None) is not None), None)
            if dated_node:
                while dated_node != tree:
                    if getattr(dated_node.up, DATE, None) is None:
                        dated_node.up.add_feature(DATE, getattr(dated_node, DATE) - dated_node.dist)
                    dated_node = dated_node.up

        trees.append(tree)
    return trees


def read_nexus(tree_path):
    with open(tree_path, 'r') as f:
        nexus = f.read()
    # replace CI_date="2019(2018,2020)" with CI_date="2018 2020"
    nexus = re.sub(r'CI_date="({})\(({}),({})\)"'.format(DATE_REGEX, DATE_REGEX, DATE_REGEX), r'CI_date="\2 \3"',
                   nexus)
    temp = tree_path + '.{}.temp'.format(datetime.timestamp(datetime.now()))
    with open(temp, 'w') as f:
        f.write(nexus)
    trees = list(Phylo.parse(temp, 'nexus'))
    os.remove(temp)
    return trees


def resolve_trees(column2states, forest):
    columns = sorted(column2states.keys())

    col2state2i = {c: dict(zip(states, range(len(states)))) for (c, states) in column2states.items()}

    def get_prediction(n):
        return '.'.join('-'.join(str(i) for i in sorted([col2state2i[c][_] for _ in getattr(n, c, set())]))
                        for c in columns)

    num_new_nodes = 0

    for tree in forest:
        todo = [tree]
        while todo:
            n = todo.pop()
            n_state = get_prediction(n)
            todo.extend(n.children)
            if len(n.children) > 2:
                state2children = defaultdict(list)
                for c in n.children:
                    state2children[get_prediction(c)].append(c)
                for state, children in state2children.items():
                    state_change = state != n_state
                    if state_change and len(children) > 1:
                        child = min(children, key=lambda _: _.dist)
                        dist = child.dist if state_change else 0
                        pol = n.add_child(dist=dist, name='{}.polytomy_{}'.format(n.name, state))
                        pol.add_feature(IS_POLYTOMY, 1)
                        pol.add_feature(DATE, getattr(child, DATE) if state_change else getattr(n, DATE))
                        pol.add_feature(DATE_CI, getattr(child, DATE_CI, None) if state_change else getattr(n, DATE_CI, None))
                        for c in columns:
                            pol.add_feature(c, getattr(child, c))
                        for c in children:
                            n.remove_child(c)
                            pol.add_child(c, dist=c.dist - dist)
                        num_new_nodes += 1
    if num_new_nodes:
        logging.getLogger('pastml').debug('Created {} new internal nodes while resolving polytomies'.format(num_new_nodes))
    else:
        logging.getLogger('pastml').debug('Could not resolve any polytomy')
    return num_new_nodes
