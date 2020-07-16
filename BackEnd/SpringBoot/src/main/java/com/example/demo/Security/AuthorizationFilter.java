package com.example.demo.Security;

import java.io.IOException;
import java.util.ArrayList;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.www.BasicAuthenticationFilter;

import io.jsonwebtoken.Jwts;

public class AuthorizationFilter extends BasicAuthenticationFilter {
    
    public AuthorizationFilter(AuthenticationManager authManager) {
        super(authManager);
     }
    
    @Override
    protected void doFilterInternal(HttpServletRequest req,
                                    HttpServletResponse res,
                                    FilterChain chain) throws IOException, ServletException {
        
    	//to get the Authorization Token from the request header(Authorization)
        String header = req.getHeader(SecurityConstant.HEADER_STRING);
        
        //to check token is Valid or not and having the default prefix value as TOKEN_PREFIX 
        if (header == null || !header.startsWith(SecurityConstant.TOKEN_PREFIX)) {
            //reads the next filter
        	chain.doFilter(req, res);
            return;
        }
        
        //to get the UsernamePasswordAuthenticationToken
        UsernamePasswordAuthenticationToken authentication = getAuthentication(req);
        
        //making the Authentication
        SecurityContextHolder.getContext().setAuthentication(authentication);
        chain.doFilter(req, res);
    }   
    
    //method to get UsernamePasswordAuthenticationToken
    private UsernamePasswordAuthenticationToken getAuthentication(HttpServletRequest request) {
    	//to get the Authorization Token from the request
    	String token = request.getHeader(SecurityConstant.HEADER_STRING);
        
        if (token != null) {
            //removing the TOKEN_PREFIX because only the token is required
            token = token.replace(SecurityConstant.TOKEN_PREFIX, "");
            //to deCrypt the token
            String user = Jwts.parser()
                    .setSigningKey( SecurityConstant.TOKEN_SECRET) //secret code used to encrypt
                    .parseClaimsJws( token )
                    .getBody()
                    .getSubject();
            
            if (user != null) {
                return new UsernamePasswordAuthenticationToken(user, null, new ArrayList<>());
            }
            
            return null;
        }
        
        return null;
    }
}
    
