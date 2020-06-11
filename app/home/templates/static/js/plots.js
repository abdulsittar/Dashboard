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
                Plotly.newPlot('linegraph', data );
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
            Plotly.newPlot('linegraph', data );
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
            $('#topicgraph').append(data.out)
        }
		});
	});
});




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