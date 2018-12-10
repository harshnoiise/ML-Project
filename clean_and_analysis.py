import json
from pprint import pprint


class Analysis(object):
    def __init__(self, path, filtered_section):
        self.generator = self.generate(path)
        self.data = self.read()
        self.filtered_section = filtered_section
    
    def generate(self, path):
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield data
    
    def read(self,):
        text = []
        try:
            while True:
                line = self.generator.__next__()
                text.append(line)
        except StopIteration:
            return text 
    
    def start(self, threshold=1000, dump=None):
        keep_class_dist = {}
        # analysis class distribution
        class_dist = {}
        altered_data = []
        
        for i, each in enumerate(self.data):
            print("article: ", i)
            section = each['section'].lower()
            keywords = each['keywords']
            headline = each['headline']
            lead_paragraph = each['lead_paragraph']
            if section in self.filtered_section:
                altered_data.append({
                'section': section,
                'keywords': keywords,
                'headline': headline,
                "lead_paragraph": lead_paragraph
                }) 
                continue 
            if ";" in section:
                tmp = [e.strip().lower() for e in section.split(";")]
                if tmp[0] == "front page" or tmp[0] == "corrections":
                    # print(section, "--", tmp[1])
                    section = tmp[1]

                else:
                    # print(section, "--", tmp[0])
                    section = tmp[0]
            if section not in class_dist:
                class_dist[section] = []

            altered_data.append({
                'section': section,
                'keywords': keywords,
                'headline': headline,
                "lead_paragraph": lead_paragraph
            }) 
            class_dist[section].append(i)
        
        for class_, idx in sorted(class_dist.items(), key=lambda x: len(x[1]), reverse=True):
            if len(class_dist[class_]) >= threshold:
                keep_class_dist[class_] = idx
        
        print("############## The number of classes is %s #################" % (len(keep_class_dist), ))
        total_num = 0
        for classes in keep_class_dist:
            total_num += len(keep_class_dist[classes])
            print("Class: %s -- Num of Instances: %s" % (classes, len(keep_class_dist[classes])))
        print("############## Total Num: %s ##################" % (total_num))
        
        new_data = []
        for classes in keep_class_dist:
            new_data += [altered_data[i] for i in keep_class_dist[classes]]
        
        if dump:
            with open(dump, 'w') as f:
                for each in new_data:
                    json.dump(each, f) 
                    f.write('\n')

    
filtered_section = [
    "nytfrontpage",
    "membercenter",
    "homepage",
    "international home",
    "insider events",
    "lesson plans",
    "letters",
    "mind",
    "move",
    "mutual funds",
    "podcasts",
    "times insider",
    "times topics",
    "women's runway",
    "world cup",
    "false"
]
analysis = Analysis('newyorktimes_filtered.jsonl', filtered_section)
analysis.start(dump="NewYorkTimesClean.jsonl")
