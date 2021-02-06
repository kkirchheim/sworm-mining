"""
https://raw.githubusercontent.com/MaksimEkin/COVID19-Literature-Clustering/
"""
from bokeh.models import CustomJS


def selection_callback(source):
    # slider call back for date selection, keyword search and cluster selection
    callback = CustomJS(args=dict(source=source), code="""
        var data = source.data; 
        var x1 = data['x1'];
        var x2 = data['x2'];
        var x1_backup = data['x1_backup'];
        var x2_backup = data['x2_backup'];
        var timestamp = data['timestamp'];
        
        var abstract = data['abstract'];
        var title = data['title'];
        var author = data['author'];
        var journal = data['journal'];
        var label = data['label'];
                
        var lower = date_range_slider.value[0];
        var upper = date_range_slider.value[1];
        var key = text_search.value;
        var clusters = cluster_choice.value;
        console.log(key);
        console.log(clusters);
        
        for (var i = 0; i < x1.length; i++) {
            var select_text = abstract[i].includes(key) 
                || title[i].includes(key) 
                || author[i].includes(key) 
                || journal[i].includes(key);
            
            var select_cluster = clusters.includes(String(label[i]));
            var select_date = timestamp[i] > lower && timestamp[i] < upper;
            
            if(select_date && select_text && select_cluster) {
                x1[i] = x1_backup[i];
                x2[i] = x2_backup[i];
            } else {
                x1[i] = undefined;
                x2[i] = undefined;
            }
        }
        source.change.emit();
        
    """)
    return callback


# handle the currently selected article
def selected_code():
    code = """
            var titles = [];
            var authors = [];
            var journals = [];
            var links = [];
            cb_data.source.selected.indices.forEach(index => titles.push(source.data['titles'][index]));
            cb_data.source.selected.indices.forEach(index => authors.push(source.data['authors'][index]));
            cb_data.source.selected.indices.forEach(index => journals.push(source.data['journal'][index]));
            cb_data.source.selected.indices.forEach(index => links.push(source.data['links'][index]));
            title = "<h4>" + titles[0].toString().replace(/<br>/g, ' ') + "</h4>";
            authors = "<p1><b>Authors:</b> " + authors[0].toString().replace(/<br>/g, ' ') + "<br>"
            // journal = "<b>Journal</b>" + journals[0].toString() + "<br>"
            link = "<b>Link:</b> <a href='" + "http://doi.org/" + links[0].toString() + "'>" + "http://doi.org/" + links[0].toString() + "</a></p1>"
            current_selection.text = title + authors + link
            current_selection.change.emit();
    """
    return code


# handle the keywords and search
def input_callback(plot, source, out_text, topics):
    # slider call back for cluster selection
    callback = CustomJS(args=dict(p=plot, source=source, out_text=out_text, topics=topics), code="""
				var key = text.value;
				key = key.toLowerCase();
				var cluster = slider.value;
                var data = source.data; 
                
                x = data['x'];
                y = data['y'];
                x_backup = data['x_backup'];
                y_backup = data['y_backup'];
                labels = data['desc'];
                abstract = data['abstract'];
                titles = data['titles'];
                authors = data['authors'];
                journal = data['journal'];
                if (cluster == '20') {
                    out_text.text = 'Keywords: Slide to specific cluster to see the keywords.';
                    for (i = 0; i < x.length; i++) {
						if(abstract[i].includes(key) || 
						titles[i].includes(key) || 
						authors[i].includes(key) || 
						journal[i].includes(key)) {
							x[i] = x_backup[i];
							y[i] = y_backup[i];
						} else {
							x[i] = undefined;
							y[i] = undefined;
						}
                    }
                }
                else {
                    out_text.text = 'Keywords: ' + topics[Number(cluster)];
                    for (i = 0; i < x.length; i++) {
                        if(labels[i] == cluster) {
							if(abstract[i].includes(key) || 
							titles[i].includes(key) || 
							authors[i].includes(key) || 
							journal[i].includes(key)) {
								x[i] = x_backup[i];
								y[i] = y_backup[i];
							} else {
								x[i] = undefined;
								y[i] = undefined;
							}
                        } else {
                            x[i] = undefined;
                            y[i] = undefined;
                        }
                    }
                }
            source.change.emit();
            """)
    return callback
