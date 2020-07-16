package com.example.demo.Security;

import java.util.Arrays;

import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import com.example.demo.Services.UserServices;



@EnableWebSecurity
public class WebSecurity extends WebSecurityConfigurerAdapter {
	private final UserServices userDetailsService;
	private final BCryptPasswordEncoder bCrytPasswordEncoder;
	
	public WebSecurity(UserServices userDetailsService,BCryptPasswordEncoder bCrytPasswordEncoder){
		this.userDetailsService=userDetailsService;
		this.bCrytPasswordEncoder=bCrytPasswordEncoder;
	}
	@Override //Entry point
	protected void configure(HttpSecurity http)throws Exception {
		http.cors().and()
		.csrf().disable().authorizeRequests()
		.antMatchers(HttpMethod.POST,SecurityConstant.SIGN_UP_URL)
		.permitAll().anyRequest().authenticated()
		.and().addFilter(getAuthenticationFilter())
		.addFilter(new AuthorizationFilter(authenticationManager()))
		.sessionManagement().sessionCreationPolicy(SessionCreationPolicy.STATELESS);	
	}
	@Override //Services used in authentication
	public void configure(AuthenticationManagerBuilder auth)throws Exception {
		auth.userDetailsService(userDetailsService).passwordEncoder(bCrytPasswordEncoder);
	}
	public AuthenticationFilter getAuthenticationFilter() throws Exception{
		final AuthenticationFilter filter =new AuthenticationFilter(authenticationManager());
		filter.setFilterProcessesUrl("/users/login");
		return filter;
	}
	@Bean
	 public CorsConfigurationSource corsConfigurationSource()
	    {
	    	final CorsConfiguration configuration = new CorsConfiguration();
	    	  
	    	configuration.setAllowedOrigins(Arrays.asList("*"));
	    	configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE","OPTIONS"));
	    	configuration.setAllowCredentials(true);
	    	configuration.setAllowedHeaders(Arrays.asList("*"));
	    	configuration.addExposedHeader("USERID");
	    	configuration.addExposedHeader("Authorization");
	    	final UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
	    	source.registerCorsConfiguration("/**", configuration);
	    	return source;
	    }
}
