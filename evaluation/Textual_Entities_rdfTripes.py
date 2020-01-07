
#External libraries and functions data input/out

from rdflib import Graph, URIRef, Literal, Namespace
from textblob import TextBlob
from get_sentence_data import *
#from Evaluations_Visual_Texual_Konwledge_v4_2 import *

#==> Prefixes to use for RDF graph (triples)
vtkel = Namespace('http://vksflickr30k.fbk.eu/resource/')
ks = Namespace('http://dkm.fbk.eu/ontologies/knowledgestore#')
owl = Namespace('http://www.w3.org/2002/07/owl#')
gaf = Namespace('http://groundedannotationframework.org/gaf#') 
nif = Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
prov = Namespace('https://www.w3.org/TR/prov-o/#prov-o-at-a-glance/')    
rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
                 
g = Graph()
g.bind("vtkel",vtkel)
g.bind("ks",ks)
g.bind("owl",owl)
g.bind("gaf",gaf)
g.bind("nif",nif)
g.bind("prov",prov)
g.bind("rdf",rdf)
#==> End of Prefixes to use for RDF graph (triples)

def Textual_Entities_rdfTripes(textual_entities,textual_entities_in_YAGO,image_id,image_captions):

    """
    This function takes textual entity mentions, YAGO types and image id and stored in RDF triple form.
    Input:
        textual_entities – textual entities mentions extracted by PIKES
        textual_entities_in_YAGO – textual entities YAGO type
        image_id – image id
        image_captions - dictionary consist of all textual annotations

    Output:
        g: RDF triples of visual and textual 

    """    
    for i in range(5):
        temp_textual_mention=''
        anchor=''
        for j in range(len(textual_entities[i])):
            uri_textual_entity=URIRef(vtkel[image_id]+'C'+str(i)+'/#'+textual_entities[i][j])
            g.add( (uri_textual_entity, URIRef(rdf['type']), URIRef(ks['TextualEntity'])) )
            g.add( (uri_textual_entity, URIRef(rdf['type']), URIRef(textual_entities_in_YAGO[i][j])) )
            anchor_words=''
            anchor=textual_entities[i][j]
            image_caption_text=TextBlob(image_captions[i]['sentence'])
            image_caption_words_list=image_caption_text.words
            anchor_count=0
            for k in range(len(image_caption_words_list)):
                if anchor in image_caption_words_list[k]:
                    anchor_count+=1
                    if anchor_count==1:
                        anchor_words=image_caption_words_list[k]

            if textual_entities[i][j][-2:-1]=='_':
                dublicate_entity_count=int(textual_entities[i][j][-1:])
                temp_textual_mention=textual_entities[i][j][:-2]
                anchor=temp_textual_mention
                image_caption_text=TextBlob(image_captions[i]['sentence'])
                image_caption_words_list=image_caption_text.words
                anchor_count=1
                start_index=0
                for k in range(len(image_caption_words_list)):
                    if anchor in image_caption_words_list[k]:
                        start_index=image_captions[i]['sentence'].find(temp_textual_mention,start_index+anchor_count)
#                        print(start_index,anchor_count)
                        anchor_count+=1
                        if anchor_count==dublicate_entity_count:
                            anchor_words=image_caption_words_list[k]
                end_index=len(anchor_words)+start_index

                uri_textual_entity_mention=URIRef(vtkel[image_id+'C'+str(i)+'/#char='+str(start_index)+','+str(end_index)])                    
                g.add( (uri_textual_entity, URIRef(gaf['denotedBy']), uri_textual_entity_mention) )  
                g.add( (uri_textual_entity_mention, URIRef(rdf['type']), URIRef(ks['TextualEntityMention'])) )
                uri_caption_id=URIRef(vtkel[image_id]+'C'+str(i)+'/')
                g.add( (uri_textual_entity_mention, URIRef(ks['mentionOf']), uri_caption_id) )
                g.add( (uri_textual_entity_mention, URIRef(prov['wasAttributedTo']), URIRef(vtkel['PikesAnnotator'])) )
                g.add( (uri_textual_entity_mention, URIRef(nif['beginIndex']), Literal(start_index)) )
                g.add( (uri_textual_entity_mention, URIRef(nif['endIndex']), Literal(end_index)) )                
                g.add( (uri_textual_entity_mention, URIRef(nif['anchorOf']), Literal(anchor_words)) )

            else:
                start_index=0
                start_index=image_captions[i]['sentence'].find(anchor_words,start_index)
                end_index=len(anchor_words)+start_index
                uri_textual_entity_mention=URIRef(vtkel[image_id+'C'+str(i)+'/#char='+str(start_index)+','+str(end_index)])                    
                g.add( (uri_textual_entity, URIRef(gaf['denotedBy']), uri_textual_entity_mention) )  
                g.add( (uri_textual_entity_mention, URIRef(rdf['type']), URIRef(ks['TextualEntityMention'])) )
                uri_caption_id=URIRef(vtkel[image_id]+'C'+str(i)+'/')
                g.add( (uri_textual_entity_mention, URIRef(ks['mentionOf']), uri_caption_id) )
                g.add( (uri_textual_entity_mention, URIRef(prov['wasAttributedTo']), URIRef(vtkel['PikesAnnotator'])) )
                g.add( (uri_textual_entity_mention, URIRef(nif['beginIndex']), Literal(start_index)) )
                g.add( (uri_textual_entity_mention, URIRef(nif['endIndex']), Literal(end_index)) )
                g.add( (uri_textual_entity_mention, URIRef(nif['anchorOf']), Literal(anchor_words)) )


#            print(uri_textual_entity,URIRef(rdf['type']),URIRef(ks['TextualEntity']))
#            print(uri_textual_entity,URIRef(rdf['type']),URIRef(textual_entities_in_YAGO[i][j]))
#            print(uri_textual_entity,URIRef(gaf['denotedBy']), uri_textual_entity_mention)
#            print(uri_textual_entity,URIRef(rdf['type']), URIRef(ks['TextualEntityMention']))
#            print(uri_textual_entity,URIRef(ks['mentionOf']), uri_caption_id)
#            print(uri_textual_entity,URIRef(nif['beginIndex']), Literal(start_index))
#            print(uri_textual_entity,URIRef(nif['endIndex']), Literal(end_index))
#            print(uri_textual_entity,URIRef(prov['wasAttributedTo']), URIRef(vtkel['PikesAnnotator']))
#            print(uri_textual_entity,URIRef(nif['anchorOf']), Literal(anchor_words))

    return g