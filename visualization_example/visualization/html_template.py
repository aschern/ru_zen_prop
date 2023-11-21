import functools
import math

HIGHLIGHT_COLORS = [
    "blue",
    "green",
    "pink",
    "orange",
    "purple",
    "teal",
    "tan",
    "red",
    "cobalt",
    "brown",
    "slate",
    "fuchsia",
    "gray",
    "blue"
]

def get_highlight_color(index):
    if index < len(HIGHLIGHT_COLORS):
        return HIGHLIGHT_COLORS[index]
    else:
        return HIGHLIGHT_COLORS[index - (len(HIGHLIGHT_COLORS) * math.floor(index / len(HIGHLIGHT_COLORS)))]

# Transofrms tokens and clusters into a tree representation
def transform_to_tree(tokens, clusters):
    def contains(span, index):
        return index >= span[0] and index <= span[1]

    inside_clusters = [{
        'cluster': -1,
        'contents': [],
        'end': -1
    }]

    for i, token in enumerate(tokens):
        # Find all the new clusters we are entering at the current index
        new_clusters = []
        for j, cluster in enumerate(clusters):
            #Make sure we're not already in this cluster
            if j not in [c['cluster'] for c in inside_clusters]:
                for span in cluster:
                    if i in span:
                        new_clusters.append({ 'end': span[1], 'cluster': j })

        # Enter each new cluster, starting with the leftmost
        new_clusters = sorted(new_clusters, key=functools.cmp_to_key(lambda a, b: b['end'] - a['end']))
        for new_cluster in new_clusters:
            #Descend into the new cluster
            inside_clusters.append({
                'cluster': new_cluster['cluster'],
                'contents': [],
                'end': new_cluster['end']
            })

        #Add the current token into the current cluster
        inside_clusters[-1]['contents'].append(token)

        # Exit each cluster we're at the end of
        while (len(inside_clusters) > 0 and inside_clusters[-1]['end'] == i):
            top_cluster = inside_clusters[-1]
            inside_clusters.pop()
            inside_clusters[-1]['contents'].append(top_cluster)

    return inside_clusters[0]['contents']


mapping = {i: el for i, el in enumerate(['Negative/Positive_concepts',
 '(PT)_Call',
 'Slogan',
 'Obfuscation,vagueness,obscurantism',
 'Consequential_Simplification',
 'Greenwashing',
 'Causal_Simplification',
 'Appeal_to_Hypocrisy',
 'Appeal_to_values',
 'Rumours',
 'Strawman',
 'Whataboutism',
 'Hate_speech,slang,name_calling',
 'Casting_Doubt',
 'Labelling',
 'Substitution_of_an_idea',
 'Statistical_deception',
 'Bluewashing',
 'Appeal_to_authority',
 'Guilt_by_Association',
 'Appeal_to_Time',
 'Flag_waving',
 '“you_should”',
 'Simplified_Interpretation',
 'Appeal_to_fear/prejudice',
 'Loaded_language',
 'Sensational_and/or_provocative_headings',
 '“I_am_like_you”',
 'Exaggeration/Minimization',
 'Red_Herring',
 'Appeal_to_popularity',
 'Repetition',
 'False_Dilemma',
 'Distraction_by_scapegoat',
 'Conversation_Killer',
 'Stereotypes'])}

#This is the function that calls itself when we recurse over the span tree.
def gen_elem(token, idx, depth, task):
    if isinstance(token, dict) or isinstance(token, list):
        if task == 'TC':
            title = mapping[token['cluster']]
        elif task == 'SI':
            title = 'PROP'
        else:
            title = token['cluster']
        return '<span key={} class="highlight {}" depth={} id={} onmouseover="handleHighlightMouseOver(this)" \
                onmouseout="handleHighlightMouseOut(this)" labelPosition="left">\
                <span class="highlight__label"><strong>{}</strong></span>\
                <span class="highlight__content">{}</span></span>'.format(idx, 
                                                                          get_highlight_color(token['cluster']), 
                                                                          depth,
                                                                          title,
                                                                          title, 
                                                                          ' '.join(span_wrapper(token['contents'], depth + 1, task)))
    else:
        return '<span>{} </span>'.format(token)
 
# Wraps the tree representation into spans indicating cluster-wise depth
def span_wrapper(tree, depth, task):
      return [gen_elem(token, idx, depth, task) for idx, token in enumerate(tree)]