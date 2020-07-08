$(function(){
    $('select#dataset').on('change', function(){
        $.ajax({
            url: "/select",
            type: "GET",
            contentType: 'application/json;charset=UTF-8',
            data: {
                'dataset': document.getElementById('dataset').value
            },
            dataType:"json",
            success: function (data) {
                $('#topicgraph').empty();
                $('#topicdist').empty();
                $('#topictable').empty();
                $('#trendtable').empty();
                $('#linegraph').empty();
            }
        });
    });
});


$(function(){
	$('#form_1').on('click', 'button',function(){
		$.ajax({
			url: '/line',
			type: 'GET',
			contentType: 'application/json;charset=UTF-8',
            data: {
                'city_news': document.getElementById('city_news').value,
                'wd_list': document.getElementById('wd_list').value
            },
			dataType:"json",
        success: function (data) {
            var layout = {
                'xaxis.range': [2010, 2015]
            }
            Plotly.newPlot('linegraph', data, layout);
        }
		});
	});
});


$(function(){
	$('#form_2').on('click', 'button',function(){
		$.ajax({
			url: '/topic',
			type: 'GET',
			contentType: 'application/json;charset=UTF-8',
            data: {
                'news_pub': document.getElementById('news_pub').value,
                'num_k': document.getElementById('num_k').value
            },
			dataType:"json",
        success: function (data) {
            $('#topicgraph').empty();
            $('#topicgraph').append(data.lda_html);
            $.ajax({
                url: '/topicdist',
                type: 'GET',
                contentType: 'application/json;charset=UTF-8',
                data: {
                    'news_pub': document.getElementById('news_pub').value,
                    'num_k': document.getElementById('num_k').value
                },
                dataType:"json",
            success: function (data) {
                Plotly.newPlot('topicdist', data);
            }
            });
        }
        });
	});
});


$(function(){
	$('div#linegraph').on('plotly_click',function(_, data){
        $.ajax({
			url: '/trendtable',
			type: 'GET',
			contentType: 'application/json;charset=UTF-8',
            data: {
                'city_news': document.getElementById('city_news').value,
                'wd_list': document.getElementById('wd_list').value,
                'year': data.points[0].x
            },
			dataType:"json",
        success: function (data) {
            $('#trendtable').empty();
            $('#trendtable').append(data.table)
        }
		});
	});
});

$(function(){
	$('div#topicdist').on('plotly_click',function(_, data){
        $.ajax({
			url: '/topicdocs',
			type: 'GET',
			contentType: 'application/json;charset=UTF-8',
            data: {
                'news_pub': document.getElementById('news_pub').value,
                'num_k': document.getElementById('num_k').value,
                'year': data.points[0].x
            },
			dataType:"json",
        success: function (data) {
            $('#topictable').empty();
            $('#topictable').append(data.table)
        }
		});
	});
});


// function bind_button() {
//     $("#annotate").on("click", function(d) {
//         var active_entity_list = $(".entity-view.active")
//         if (active_entity_list.length != 1) {
//             alert("Please select an item")
//         }
//         var active_entity = active_entity_list.first()
//
//         // get the raw text and final annotations
//         var article = $("img").attr("src").split("/")[2];
//         var url = "annotate/" + article + "/" + active_entity.attr("data-src");
//         console.log("Calling: " + url);
//         $.ajax({
//             type: "GET",
//             url: url,
//             success: function(response) {
//                 set_page(response);
//                 bind_button();
//                 console.log("SUCESS!");
//             },
//             error: function(error_response) {
//                 console.error(error_response);
//             }
//         });
//     });
// }


//$('#first_cat').on('change',function(){
//    $.ajax({
//        url: "/bar",
//        type: "GET",
//        contentType: 'application/json;charset=UTF-8',
//        data: {
//            'selected': document.getElementById('first_cat').value
//
//        },
//        dataType:"json",
//        success: function (data) {
//            Plotly.newPlot('bargraph', data );
//        }
//    });
//});