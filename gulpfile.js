var gulp = require('gulp'),
    FOLDER = 'bower_components';

gulp.task('materialize:css', function(){
  return gulp.src(FOLDER + '/materialize/dist/css/materialize.min.css')
    .pipe(gulp.dest('static/styles'))
});

gulp.task('materialize:js', function(){
  return gulp.src(FOLDER + '/materialize/dist/js/materialize.min.js')
    .pipe(gulp.dest('static/scripts'))
});

gulp.task('materialize', ['materialize:css', 'materialize:js']);

gulp.task('jquery', function(){
  return gulp.src([FOLDER + '/jquery/dist/jquery.min.js',
                   FOLDER + '/jquery/dist/jquery.min.map'])
    .pipe(gulp.dest('static/scripts'))
});

gulp.task('typed', function(){
  return gulp.src([FOLDER + '/typed.js/lib/typed.min.js',
                   FOLDER + '/typed.js/lib/typed.min.js.map'])
    .pipe(gulp.dest('static/scripts'))
});

gulp.task('jqval', function() {
  return gulp.src([FOLDER + '/jquery-validation/dist/jquery.validate.min.js',
                   FOLDER + '/jquery-validation/dist/additional-methods.min.js'])
    .pipe(gulp.dest('static/scripts'))
})

gulp.task('default', [ 'materialize', 'jquery', 'jqval', 'typed']);
