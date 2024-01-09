import logging
from collections import defaultdict
from pastml.tree import IS_POLYTOMY, copy_forest

import numpy as np

VERTICAL = 'VERTICAL'
HORIZONTAL = 'HORIZONTAL'
TRIM = 'TRIM'

IS_TIP = 'is_tip'

REASONABLE_NUMBER_OF_TIPS = 15

CATEGORIES = 'categories'

NUM_TIPS_INSIDE = 'max_size'

TIPS_INSIDE = 'in_tips'
INTERNAL_NODES_INSIDE = 'in_ns'
TIPS_BELOW = 'all_tips'
ROOTS = 'roots'
ROOT_DATES = 'root_dates'

COMPRESSED_NODE = 'compressed_node'

METACHILD = 'metachild'

IN_FOCUS = 'in_focus'
AROUND_FOCUS = 'around_focus'
UP_FOCUS = 'up_focus'


def _tree2pajek_vertices_arcs(compressed_tree, nodes, edges, columns):
    n2id = {}

    def get_states(n, columns):
        res = []
        for column in columns:
            values = n.props.get(column, set())
            value = values if isinstance(values, str) else ' or '.join(sorted(values))
            res.append('{}:{}'.format(column, value))
        return res

    for id, n in enumerate(compressed_tree.traverse('preorder'), start=len(nodes) + 1):
        n2id[n] = id
        if not n.is_root:
            edges.append('{} {} {}'.format(n2id[n.up], id, len(n.props.get(ROOTS))))
        nodes.append('{} "{}" "{}" {}'.format(id, n.name,
                                              (';'.join(','.join(_.name for _ in ti) for ti in n.props.get(TIPS_INSIDE))),
                                              ' '.join('"{}"'.format(_) for _ in get_states(n, columns))))


def save_to_pajek(nodes, edges, pajek):
    """
    Saves a compressed tree into Pajek format:

    *vertices <number_of_vertices>
    <id_1> "<vertex_name>" "<tips_inside>" "<column1>:<state(s)>" ["<column2>:<state(s)>" ...]
    ...
    *arcs
    <source_id> <target_id> <weight>
    ...

    <tips_inside> list tips that were vertically compressed inside this node: they are comma-separated.
    If the node was also horizontally merged (i.e. represents several similar configurations),
    the tip sets corresponding to different configurations are semicolon-separated.

    <state(s)> lists the states predicted for the corresponding column:
    if there are several states, they are separated with " or ".


    :return: void (creates a file specified in the pajek argument)
    """
    with open(pajek, 'w+') as f:
        f.write('*vertices {}\n'.format(len(nodes)))
        f.write('\n'.join(nodes))
        f.write('\n')
        f.write('*arcs\n')
        f.write('\n'.join(edges))


def compress_tree(tree, columns, can_merge_diff_sizes=True, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS, mixed=False,
                  pajek=None, pajek_timing=VERTICAL):
    compressed_tree = copy_forest([tree], features=columns | set(tree.props))[0]

    for n_compressed, n in zip(compressed_tree.traverse('postorder'), tree.traverse('postorder')):
        n_compressed.add_prop(TIPS_BELOW, [list(n_compressed.leaves())])
        n_compressed.add_prop(TIPS_INSIDE, [])
        n_compressed.add_prop(INTERNAL_NODES_INSIDE, [])
        n_compressed.add_prop(ROOTS, [n])
        if n_compressed.is_leaf:
            n_compressed.props.get(TIPS_INSIDE).append(n)
        elif not  n_compressed.props.get(IS_POLYTOMY, False):
             n_compressed.props.get(INTERNAL_NODES_INSIDE).append(n)
        n.add_prop(COMPRESSED_NODE, n_compressed)

    collapse_vertically(compressed_tree, columns, mixed=mixed)
    if pajek is not None and VERTICAL == pajek_timing:
        _tree2pajek_vertices_arcs(compressed_tree, *pajek, columns=sorted(columns))

    for n in compressed_tree.traverse():
        n.add_prop(NUM_TIPS_INSIDE, len(n.props.get(TIPS_INSIDE)))
        n.add_prop(TIPS_INSIDE, [n.props.get(TIPS_INSIDE)])
        n.add_prop(INTERNAL_NODES_INSIDE, [n.props.get(INTERNAL_NODES_INSIDE)])

    get_bin = lambda _: _
    collapse_horizontally(compressed_tree, columns, get_bin, mixed=mixed)

    if can_merge_diff_sizes and len(compressed_tree) > tip_size_threshold:
        get_bin = lambda _: int(np.log10(max(1, _)))
        logging.getLogger('pastml').debug('Allowed merging nodes of different sizes.')
        collapse_horizontally(compressed_tree, columns, get_bin, mixed=mixed)

    if pajek is not None and HORIZONTAL == pajek_timing:
        _tree2pajek_vertices_arcs(compressed_tree, *pajek, columns=sorted(columns))

    if len(compressed_tree) > tip_size_threshold:
        for n in compressed_tree.traverse('preorder'):
            multiplier = (n.up.props.get('multiplier') if n.up else 1) * len(n.props.get(ROOTS))
            n.add_prop('multiplier', multiplier)

        def get_tsize(n):
            if n.props.get(IN_FOCUS, False) or n.props.get(AROUND_FOCUS, False) or n.props.get(UP_FOCUS, False):
                return np.inf
            return n.props.get(NUM_TIPS_INSIDE) * n.props.get('multiplier')

        node_thresholds = []
        for n in compressed_tree.traverse('postorder'):
            children_bs = 0 if not n.children else max(get_tsize(_) for _ in n.children)
            bs = get_tsize(n)
            # if bs > children_bs it means that the trimming threshold for the node is higher
            # than the ones for its children
            if not n.is_root and bs > children_bs:
                node_thresholds.append(bs)
        threshold = sorted(node_thresholds)[-tip_size_threshold]

        if min(node_thresholds) >= threshold:
            if threshold == np.inf:
                logging.getLogger('pastml') .debug('All tips are in focus.')
            else:
                logging.getLogger('pastml')\
                    .debug('No tip is smaller than the threshold ({}, the size of the {}-th largest tip).'
                           .format(threshold, tip_size_threshold))
        else:
            if threshold == np.inf:
                logging.getLogger('pastml')\
                    .debug('Removing all the out of focus tips (as there are at least {} tips in focus).'
                           .format(tip_size_threshold))
            else:
                logging.getLogger('pastml').debug('Set tip size threshold to {} (the size of the {}-th largest tip).'
                                                  .format(threshold, tip_size_threshold))
            remove_small_tips(compressed_tree=compressed_tree, full_tree=tree,
                              to_be_removed=lambda _: get_tsize(_) < threshold)
            remove_mediators(compressed_tree, columns)
            collapse_horizontally(compressed_tree, columns, get_bin, mixed=mixed)

    if pajek is not None and TRIM == pajek_timing:
        _tree2pajek_vertices_arcs(compressed_tree, *pajek, columns=sorted(columns))

    return compressed_tree


def collapse_horizontally(tree, columns, tips2bin, mixed=False):
    config_cache = {}

    def get_configuration(n):
        if n.name not in config_cache:
            # Configuration is (branch_width, (size, states, child_configurations)),
            # where branch_width is only used for recursive calls and is ignored when considering a merge
            config_cache[n.name] = (len(n.props.get(TIPS_INSIDE)),
                               (tips2bin(n.props.get(NUM_TIPS_INSIDE)),
                                tuple(tuple(sorted(n.props.get(column, set()))) for column in columns),
                                tuple(sorted([get_configuration(_) for _ in n.children]))))
        return config_cache[n.name]

    collapsed_configurations = 0

    uncompressable_ids = set()
    for n in tree.traverse('postorder'):
        config2children = defaultdict(list)
        for _ in n.children:
            if mixed and (_.props.get(IN_FOCUS, False) or _.name in uncompressable_ids):
                uncompressable_ids.add(_.name)
                uncompressable_ids.add(n.name)
            else:
                # use (size, states, child_configurations) as configuration (ignore branch width)
                config2children[get_configuration(_)[1]].append(_)
        for children in (_ for _ in config2children.values() if len(_) > 1):
            collapsed_configurations += 1
            child = children[0]
            for sibling in children[1:]:
                child.props.get(TIPS_INSIDE).extend(sibling.props.get(TIPS_INSIDE))
                for ti in sibling.props.get(TIPS_INSIDE):
                    for _ in ti:
                        _.add_prop(COMPRESSED_NODE, child)
                child.props.get(INTERNAL_NODES_INSIDE).extend(sibling.props.get(INTERNAL_NODES_INSIDE))
                for ii in sibling.props.get(INTERNAL_NODES_INSIDE):
                    for _ in ii:
                        _.add_prop(COMPRESSED_NODE, child)
                child.props.get(ROOTS).extend(sibling.props.get(ROOTS))
                child.props.get(TIPS_BELOW).extend(sibling.props.get(TIPS_BELOW))
                n.remove_child(sibling)
            child.add_prop(METACHILD, True)
            child.add_prop(NUM_TIPS_INSIDE,
                              sum(len(_) for _ in child.props.get(TIPS_INSIDE)) / len(child.props.get(TIPS_INSIDE)))
            if child.name in config_cache:
                config_cache[child.name] = (len(child.props.get(TIPS_INSIDE)), config_cache[child.name][1])
    if collapsed_configurations:
        logging.getLogger('pastml').debug(
            'Collapsed {} sets of equivalent configurations horizontally.'.format(collapsed_configurations))


def remove_small_tips(compressed_tree, full_tree, to_be_removed):
    num_removed = 0
    changed = True
    while changed:
        changed = False
        for l in compressed_tree.get_leaves():
            parent = l.up
            if parent and to_be_removed(l):
                num_removed += 1
                parent.remove_child(l)
                # remove the corresponding nodes from the non-collapsed tree
                for ti in l.props.get(TIPS_INSIDE):
                    for _ in ti:
                        _.up.remove_child(_)
                for ii in l.props.get(INTERNAL_NODES_INSIDE):
                    for _ in ii:
                        _.up.remove_child(_)
                changed = True

    # if the full tree now contains non-sampled tips,
    # remove them from the tree and from the corresponding collapsed nodes
    todo = list(full_tree)
    while todo:
        t = todo.pop()
        if not t.props.get(IS_TIP, False):
            parent = t.up
            t.up.remove_child(t)
            if parent.is_leaf:
                todo.append(parent)
            for ini_list in t.props.get(COMPRESSED_NODE).props.get(INTERNAL_NODES_INSIDE):
                if t in ini_list:
                    ini_list.remove(t)

    logging.getLogger('pastml').debug(
        'Recursively removed {} tips of size smaller than the threshold.'.format(num_removed))


def collapse_vertically(tree, columns, mixed=False):
    """
    Collapses a child node into its parent if they are in the same state.
    :param columns: a list of characters
    :param tree: ete3.Tree
    :param mixed: if True then the nodes in focus will not get collapsed
    :return: void, modifies the input tree
    """

    def _same_states(node1, node2, columns):
        for column in columns:
            if node1.props.get(column, set()) != node2.props.get(column, set()):
                return False
        if mixed:
            if node1.props.get(IN_FOCUS, False) or node2.props.get(IN_FOCUS, False):
                return False
            if node1.props.get(UP_FOCUS, False) and not node2.props.get(IN_FOCUS, False) and not node2.props.get(UP_FOCUS, False):
                node2.add_prop(AROUND_FOCUS, True)
                return False
            if node2.props.get(UP_FOCUS, False) and not node1.props.get(IN_FOCUS, False) and not node1.props.get(UP_FOCUS, False):
                node1.add_prop(AROUND_FOCUS, True)
                return False
        return True

    num_collapsed = 0
    for n in tree.traverse('postorder'):
        if n.is_leaf:
            continue

        children = list(n.children)
        for child in children:
            # merge the child into this node if their states are the same
            if _same_states(n, child, columns):
                n.props.get(TIPS_INSIDE).extend(child.props.get(TIPS_INSIDE))
                for _ in child.props.get(TIPS_INSIDE):
                    _.add_prop(COMPRESSED_NODE, n)
                n.props.get(INTERNAL_NODES_INSIDE).extend(child.props.get(INTERNAL_NODES_INSIDE))
                for _ in child.props.get(INTERNAL_NODES_INSIDE):
                    _.add_prop(COMPRESSED_NODE, n)

                n.remove_child(child)
                grandchildren = list(child.children)
                for grandchild in grandchildren:
                    n.add_child(grandchild)
                num_collapsed += 1
    if num_collapsed:
        logging.getLogger('pastml').debug('Collapsed vertically {} internal nodes without state change.'
                                          .format(num_collapsed))


def remove_mediators(tree, columns):
    """
    Removes intermediate nodes that are just mediators between their parent and child states.
    :param columns: list of characters
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """
    num_removed = 0
    for n in tree.traverse('postorder'):
        if n.props.get(METACHILD, False) or n.is_leaf or len(n.children) > 1 or n.is_root \
                or n.props.get(NUM_TIPS_INSIDE) > 0:
            continue

        parent = n.up
        child = n.children[0]

        compatible = True
        for column in columns:
            states = n.props.get(column, set())
            parent_states = parent.props.get(column, set())
            child_states = child.props.get(column, set())
            # if mediator has unresolved states, it should hesitate between the parent and the child:
            if len(states) < 2 or states != child_states | parent_states:
                compatible = False
                break

        if compatible:
            parent.remove_child(n)
            parent.add_child(child)
            # update the uncompressed tree
            for ii in n.props.get(INTERNAL_NODES_INSIDE):
                for _ in ii:
                    for c in list(_.children):
                        _.up.add_child(c)
                    _.up.remove_child(_)
            num_removed += 1
    if num_removed:
        logging.getLogger('pastml').debug("Removed {} internal node{}"
                                          " with the state unresolved between the parent's and the only child's."
                                          .format(num_removed, '' if num_removed == 1 else 's'))
