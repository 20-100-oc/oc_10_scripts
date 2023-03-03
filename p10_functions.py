import numpy as np



def get_turn_entities(data, index, ls_entities):
    luis_data = []

    for conversation in data['turns'][index]:
        json_part = {}
        txt = conversation['text'].lower()
        json_part['text'] = txt
        json_part['intentName'] = 'BookFlight'

        if conversation['author'] == 'user':
            for act in conversation['labels']['acts']:
                entities = []
                for arg in act['args']:
                    if arg['key'] in ls_entities:
                        entity = {}
                        key = arg['key'].lower()
                        if 'val' in arg.keys():
                            val = arg['val'].lower()
                            if val != '-1':
                                startCharIndex = txt.index(val)
                                endCharIndex = startCharIndex + len(val)
                                entity['entityName'] = key
                                entity['startCharIndex'] = startCharIndex
                                entity['endCharIndex'] = endCharIndex
                                entities.append(entity)
                json_part['entityLabels'] = entities
                
        if len(json_part) > 0:
            if 'entityLabels' in json_part.keys():
                if len(json_part['entityLabels'])>0:
                    luis_data.append(json_part)
    
    return luis_data



def convert_data(data, ls_entities):
    luis_data = []
    for i in range(data.shape[0]):
        json_part = get_turn_entities(data, i, ls_entities)
        if len(json_part)>0:
            for j in range(len(json_part)):
                luis_data.append(json_part[j])
    return luis_data



def create_train_test_sets(val_set_size, luis_data):
    indices = np.arange(len(luis_data))
    shuffled_indices = np.random.permutation(indices)
    train_indices = shuffled_indices[val_set_size:]
    val_indices = shuffled_indices[:val_set_size]
    
    train_set = [luis_data[i] for i in train_indices]

    val_set = []
    for i in val_indices:
        y = {
            'entities': luis_data[i]['entityLabels'],
            'intent': luis_data[i]['intentName'], 
            'text': luis_data[i]['text']
        }
        val_set.append(y)

    return train_set, val_set
