function boxes()
{
    $('#boxes').css("display","block");
    $('#home_info').css("display","none");
}
function blue(str)
{
    $('#' + 'Mining_Tool').css("background-color", "#ffffff");
    $('#' + 'Ball_Bearing').css("background-color", "#ffffff");
    $('#' + 'Spur_Gear').css("background-color", "#ffffff");
    $('#' + 'Turning_Tool').css("background-color", "#ffffff");
    $('#' + str).css("background-color", "a8d1ee");
    boxes(str);
}